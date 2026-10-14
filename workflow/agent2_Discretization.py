# -*- coding: utf-8 -*-
import os
import sys
import logging
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import glob
import argparse
import json
import copy
# 导入标准的 OpenAI 库
from openai import OpenAI
from parameter_contract import (
    PARAMETER_CONTRACT_SCHEMA_GUIDANCE,
    ParameterContractError,
    bind_immutable_run_inputs,
    extract_parameter_contract,
    format_parameter_contract,
    load_parameter_contract,
    save_parameter_contract,
    validate_parameter_contract,
)
from run_spec import load_run_spec
from audit_report_support import publish_audit_report
from machine_contracts import (
    extract_machine_contract,
    extract_json_contract_candidate,
    format_machine_contract,
    load_machine_contract,
    save_machine_contract,
    validate_discretization_contract,
    validate_theory_contract,
)
from pipeline_support import (
    atomic_json_write,
    categorized_output_directory,
    resolve_output_tag,
    write_stage_output_manifest,
)
from review_protocol import (
    AUTOMATIC_REVISION_POLICY_VERSION,
    automatic_revision_count,
    extract_review_decision,
    next_review_paths,
    render_review_audit,
)
from vqe_experience_store import (
    format_discretization_experience_context,
    summarize_discretization_problem,
)
from model_output_ingestion import json_object_candidates

# ==========================================
# 通用日志与辅助函数
# ==========================================
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def extract_latex(text):
    tb = chr(96) * 3
    pattern = re.compile(f'{tb}(?:latex|tex)?\\s*(.*?)\\s*{tb}', re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def drop_observable_only_parameters(parameter_contract):
    """Keep observable controls in the observable specification, their sole owner."""
    excluded = {"requested_components"}
    parameter_contract["parameters"] = [
        item for item in parameter_contract.get("parameters", [])
        if item.get("name") not in excluded
    ]
    for discretization in parameter_contract.get("discretizations", []):
        if isinstance(discretization.get("parameter_references"), list):
            discretization["parameter_references"] = [
                name for name in discretization["parameter_references"]
                if name not in excluded
            ]
    for constraint in parameter_contract.get("constraints", []):
        if isinstance(constraint.get("parameters"), list):
            constraint["parameters"] = [
                name for name in constraint["parameters"] if name not in excluded
            ]
    return validate_parameter_contract(parameter_contract)


def next_reviewer_attempt_paths(audit_directory, output_tag):
    return next_review_paths(audit_directory, "Agent2", output_tag)


def reusable_or_new_draft_paths(report_directory, contract_directory, output_tag):
    """Return one complete checkpoint, or a fresh non-overwriting checkpoint set."""
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_Attempt{}".format(attempt)
        paths = (
            os.path.join(
                report_directory, "Agent2_Draft_{}{}.tex".format(output_tag, suffix)
            ),
            os.path.join(
                contract_directory,
                "Agent2_Draft_ParameterContract_{}{}.json".format(output_tag, suffix),
            ),
            os.path.join(
                contract_directory,
                "Agent2_Draft_DiscretizationContract_{}{}.json".format(
                    output_tag, suffix
                ),
            ),
        )
        existence = [os.path.isfile(path) for path in paths]
        if all(existence) or not any(existence):
            return paths
        attempt += 1


def next_generator_raw_path(audit_directory, output_tag):
    """Allocate a non-overwriting raw-response path before local validation."""
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else "_Attempt{}".format(attempt)
        path = os.path.join(
            audit_directory,
            "Agent2_GeneratorRaw_{}{}.txt".format(output_tag, suffix),
        )
        if not os.path.exists(path):
            return path
        attempt += 1


def latest_generator_raw_path(audit_directory, output_tag):
    paths = glob.glob(os.path.join(
        audit_directory, "Agent2_GeneratorRaw_{}*.txt".format(output_tag)
    ))
    return max(paths, key=os.path.getmtime) if paths else None


def extract_discretization_response(text, theory_contract, run_spec):
    parameter_contract = extract_parameter_contract(text)
    parameter_contract["schema_version"] = "1.1"
    parameter_contract = bind_immutable_run_inputs(
        parameter_contract,
        run_spec["user_input"]["parameters"],
        run_spec["user_input"]["external_states"],
    )
    parameter_contract = drop_observable_only_parameters(parameter_contract)
    # External-state identity and transport metadata are controller-owned, not
    # prose for Agent2 or its reviewer to regenerate.
    parameter_contract["external_states"] = copy.deepcopy(
        theory_contract["external_states"]
    )
    discretization_contract = extract_json_contract_candidate(
        text, "lattice_discretization"
    )
    discretization_contract["schema_version"] = "1.0"
    discretization_contract["contract_type"] = "lattice_discretization"
    discretization_contract.pop("theory_contract_sha256", None)
    discretization_contract.pop("parameter_contract_sha256", None)
    discretization_contract.pop("external_states", None)
    discretization_contract.pop("validation_obligations", None)
    return (
        extract_latex(text),
        parameter_contract,
        validate_discretization_contract(
            discretization_contract,
            expected_theory_contract=theory_contract,
            expected_parameter_contract=parameter_contract,
        ),
    )


def _merge_entries_by_key(base, corrections, keys):
    """Replace only correction entries identified by an existing natural key."""
    merged = copy.deepcopy(base) if isinstance(base, list) else []
    if not isinstance(corrections, list):
        return merged
    for correction in corrections:
        if not isinstance(correction, dict):
            continue
        identity_key = next(
            (key for key in keys if correction.get(key) is not None), None
        )
        match = next((
            index for index, existing in enumerate(merged)
            if isinstance(existing, dict)
            and identity_key is not None
            and existing.get(identity_key) == correction.get(identity_key)
        ), None)
        if match is None:
            merged.append(copy.deepcopy(correction))
        else:
            merged[match] = copy.deepcopy(correction)
    return merged


def apply_agent2_correction(
    text, parameter_contract, discretization_contract, theory_contract, run_spec
):
    """Merge one bounded Agent2 correction and validate the resulting pair."""
    candidates = [
        item for item in json_object_candidates(text)
        if "parameter_contract" in item or "discretization_contract" in item
    ]
    if len(candidates) != 1:
        raise ParameterContractError(
            "Expected exactly one Agent2 correction object; found {}.".format(
                len(candidates)
            )
        )
    correction = candidates[0]
    parameter = copy.deepcopy(parameter_contract)
    parameter_changes = correction.get("parameter_contract", {})
    if not isinstance(parameter_changes, dict):
        raise ParameterContractError("parameter_contract correction must be an object.")
    for key, identities in (
        ("parameters", ("name",)),
        ("constraints", ("id",)),
        ("validation_obligations", ("id",)),
        ("discretizations", ("sector", "scheme")),
    ):
        if key in parameter_changes:
            parameter[key] = _merge_entries_by_key(
                parameter.get(key), parameter_changes[key], identities
            )
    if isinstance(parameter_changes.get("model"), dict):
        parameter.setdefault("model", {}).update(parameter_changes["model"])
    parameter = bind_immutable_run_inputs(
        parameter,
        run_spec["user_input"]["parameters"],
        run_spec["user_input"]["external_states"],
    )
    parameter = drop_observable_only_parameters(parameter)
    parameter["external_states"] = copy.deepcopy(theory_contract["external_states"])
    parameter = validate_parameter_contract(parameter)

    discretization = copy.deepcopy(discretization_contract)
    discretization_changes = correction.get("discretization_contract", {})
    if not isinstance(discretization_changes, dict):
        raise ParameterContractError(
            "discretization_contract correction must be an object."
        )
    for key, identities in (
        ("definitions", ("id",)),
        ("field_encodings", ("sector", "scheme")),
        ("lattice_symmetries", ("id",)),
    ):
        if key in discretization_changes:
            discretization[key] = _merge_entries_by_key(
                discretization.get(key), discretization_changes[key], identities
            )
    if isinstance(discretization_changes.get("lattice"), dict):
        discretization.setdefault("lattice", {}).update(
            discretization_changes["lattice"]
        )
    discretization = validate_discretization_contract(
        discretization,
        expected_theory_contract=theory_contract,
        expected_parameter_contract=parameter,
    )
    return parameter, discretization


def required_run_parameter_findings(parameter_contract):
    """Return deterministic fatal findings before a reviewer can accept."""
    unresolved = [
        str(item.get("name"))
        for item in parameter_contract.get("parameters", [])
        if item.get("required_for_run") is True
        and item.get("selected_value") is None
    ]
    if not unresolved:
        return []
    return [{
        "severity": "fatal",
        "message": (
            "The contract is not executable because required_for_run parameters "
            "have null selected_value: {}. Resolve quantities determined by the "
            "RunSpec coordinate convention or a justified discretization choice. "
            "A mathematical path/pullback/operator prescription is a contract "
            "definition or validation obligation, not an independently assigned "
            "runtime parameter."
        ).format(", ".join(unresolved)),
    }]


# ==========================================
# arXiv 专属检索工具
# ==========================================

def search_arxiv(query_keywords, max_results=3):
    """
    使用 arXiv 官方 API 检索物理论文摘要
    """
    logger = logging.getLogger("ArXiv_Searcher")
    logger.info(f"正在 arXiv 检索相关文献: {query_keywords}")
    
    search_query = urllib.parse.quote(query_keywords)
    url = f'http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    
    try:
        # 增加请求头伪装成浏览器
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        response = urllib.request.urlopen(req)
        
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            # 修复了原来代码中 replac/e 的拼写错误
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            published = entry.find('atom:published', ns).text
            papers.append(f"Title: {title}\nPublished: {published}\nAbstract: {summary}\n")
            
        if not papers:
            logger.warning("未能在 arXiv 找到高度相关的文献，将回退到大模型的基础知识。")
            return "No recent relevant papers found."
            
        logger.info(f"成功获取 {len(papers)} 篇最新 arXiv 论文摘要。")
        return "\n---\n".join(papers)
    except Exception as e:
        logger.error(f"arXiv 检索失败: {e}")
        return f"ArXiv search failed: {e}"


def extract_relevant_sections_from_library(
    library_path="A2_library", target_keywords=None, max_chars=12000, primary_keywords=None
):
    """
    扫描 A2_library/ 目录下的所有 txt 文件，提取包含目标关键词的 LaTeX section 或段落。
    max_chars 限制最终输出的字符数，防止 Token 爆炸（12000字符约等于 3000 Token）。
    """
    logger = setup_logger("Local_Library_Searcher")
    
    if target_keywords is None:
        target_keywords = ["Hamiltonian", "discretization", "lattice", "truncation", 
                           "gauge field", "electric field", "fermion", "link operator", 
                           "Kogut-Susskind", "Pauli"]
    primary_keywords_lower = {
        keyword.lower() for keyword in (primary_keywords or []) if keyword.strip()
    }
                           
    txt_files = glob.glob(os.path.join(library_path, "**", "*.txt"), recursive=True)
    tex_files = glob.glob(os.path.join(library_path, "**", "*.tex"), recursive=True)
    
    paper_files = txt_files + tex_files
    
    if not paper_files:
        logger.warning(f"在 {library_path} 及其子目录中没有找到任何 .txt 或 .tex 文献。")
        return None

    logger.info(f"在本地文献库找到 {len(paper_files)} 篇文章，开始检索相关 Section...")

    scored_chunks = []

    def split_large_chunk(chunk, chunk_limit=6000):
        """将超长 section 继续按段落拆分，避免单个大块挤占全部上下文。"""
        if len(chunk) <= chunk_limit:
            return [chunk]

        pieces = []
        current_piece = ""
        for paragraph in re.split(r'\n\s*\n', chunk):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # 极长的单段（例如生成的表格）仍需硬切分，保证后续能够装入上下文。
            if len(paragraph) > chunk_limit:
                if current_piece:
                    pieces.append(current_piece)
                    current_piece = ""
                pieces.extend(
                    paragraph[start:start + chunk_limit]
                    for start in range(0, len(paragraph), chunk_limit)
                )
                continue

            candidate = f"{current_piece}\n\n{paragraph}" if current_piece else paragraph
            if len(candidate) > chunk_limit:
                pieces.append(current_piece)
                current_piece = paragraph
            else:
                current_piece = candidate

        if current_piece:
            pieces.append(current_piece)
        return pieces

    for file_path in paper_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
            continue

        if r"\section" in text or r"\subsection" in text:
            chunks = re.split(r'(?=\\(?:section|subsection)\*?\s*\{)', text)
        else:
            chunks = text.split('\n\n')

        for section_chunk in chunks:
            section_match = re.search(
                r'\\(?:section|subsection)\*?\s*\{([^{}]*)\}',
                section_chunk[:1000],
            )
            section_name = section_match.group(1).strip() if section_match else "Document preamble"
            for chunk in split_large_chunk(section_chunk):
                chunk = chunk.strip()
                if not chunk:
                    continue

                chunk_lower = chunk.lower()
                weighted_hits = 0.0
                for kw in target_keywords:
                    count = chunk_lower.count(kw.lower())
                    # 多词术语通常比单个通用词更能代表目标模型与具体算符。
                    if kw.lower() in primary_keywords_lower:
                        weight = 5.0
                    else:
                        weight = 2.0 if " " in kw.strip() else 1.0
                    weighted_hits += count * weight

                if weighted_hits > 0:
                    formula_markers = (
                        r"\begin{equation", r"\begin{align", r"\begin{gather", r"\[", "$$"
                    )
                    formula_bonus = 2.0 if any(marker in chunk for marker in formula_markers) else 0.0
                    # 使用温和的长度归一化，避免论文前言或超长附录仅凭篇幅胜出。
                    score = (weighted_hits + formula_bonus) / max(1.0, len(chunk) / 3000.0) ** 0.5
                    if section_name == "Document preamble":
                        score *= 0.5
                    source_name = os.path.relpath(file_path, library_path)
                    scored_chunks.append((score, source_name, section_name, chunk))

    if not scored_chunks:
        logger.warning("本地文献库中未找到与目标任务相关的章节。")
        return None

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    header = "### EXTRACTED RELEVANT SECTIONS FROM LOCAL LIBRARY ###\n\n"
    assembled_text = header
    current_chars = len(assembled_text)
    selected_chunks = 0
    chunks_per_file = {}
    selected_fingerprints = set()
    
    for score, filename, section_name, chunk in scored_chunks:
        # 防止某一篇论文独占检索上下文，让 Agent 2 能比较多个来源。
        if chunks_per_file.get(filename, 0) >= 3:
            continue

        # 本地库可能同时包含完整论文和由该论文摘出的 target_paper；避免重复占用上下文。
        fingerprint = re.sub(r'\s+', ' ', chunk).strip().lower()[:1000]
        if fingerprint in selected_fingerprints:
            continue

        addition = (
            f"--- From Paper: {filename}; Section: {section_name} "
            f"(Relevance Score: {score:.2f}) ---\n{chunk}\n\n"
        )
        if current_chars + len(addition) > max_chars:
            remaining_chars = max_chars - current_chars
            if remaining_chars > 0:
                assembled_text += addition[:remaining_chars]
                current_chars += remaining_chars
                selected_chunks += 1
                chunks_per_file[filename] = chunks_per_file.get(filename, 0) + 1
                selected_fingerprints.add(fingerprint)
            logger.info("已达到设定的上下文长度上限，停止提取。")
            break
        assembled_text += addition
        current_chars += len(addition)
        selected_chunks += 1
        chunks_per_file[filename] = chunks_per_file.get(filename, 0) + 1
        selected_fingerprints.add(fingerprint)

    if current_chars == len(header):
        logger.warning("找到了相关章节，但没有成功装入任何正文，将回退到备用检索。")
        return None

    logger.info(f"成功提取了 {selected_chunks} 个高相关度片段，总字符数: {current_chars}")
    return assembled_text

# ==========================================
# Agent 2: 格点离散化智能体
# ==========================================
class LatticeDiscretizationAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol", max_retries=2):
        # 使用 OpenAI 客户端，支持 base_url 路由
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_retries = max_retries
        self.logger = setup_logger("Agent2_LatticeDiscretization")
        
    def _call_model_with_retry(self, prompt, system_instruction):
        """处理大模型的重试逻辑与消息构建 (兼容 OpenAI 格式)"""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.max_retries):
            try:
                # 移除了 temperature，解锁推理模型的完整能力
                response_stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True
                )
                full_content = ""
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
            except Exception as e:
                self.logger.warning(f"网络或服务器异常 (可能是 503 拥挤): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = 5 * (attempt + 1)
                    self.logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error("达到最大重试次数，调用失败。")
                    sys.exit(1)

    def process_continuous_theory(
        self, agent1_report, theory_contract, run_spec, physics_target, simulation_parameters,
        full_paper_text=None, raw_response_path=None,
        discretization_experience=None,
    ):
        self.logger.info("接收到 Agent 1 的连续场论报告，开始进行格点离散化和数字化方案设计...")
        
        tb = chr(96) * 3 
        discretization_example = {
            "schema_version": "1.0",
            "contract_type": "lattice_discretization",
            "experience_selection": {
                "decision": "reuse, deviate, or none",
                "source_record_id": "historical database record ID or null",
                "reason": "specific compatibility decision",
            },
            "lattice": {
                "geometry": "exact dimension, size and indexing",
                "spatial_boundary_condition": "exact boundary condition",
            },
            "field_encodings": [
                {
                    "sector": "applicable field sector",
                    "scheme": "exact discretization scheme",
                    "degrees_of_freedom": "exact finite lattice degrees of freedom",
                    "truncation": "exact truncation or explained none",
                }
            ],
            "definitions": [
                {
                    "id": "lattice_hamiltonian",
                    "kind": "lattice_hamiltonian",
                    "canonical_expression": "exact indexed machine-readable expression",
                    "display_latex": "exact LaTeX expression",
                    "dependencies": [],
                },
                {
                    "id": "lattice_observable",
                    "kind": "lattice_observable",
                    "canonical_expression": "exact ordered lattice observable",
                    "display_latex": "exact LaTeX expression",
                    "dependencies": [],
                },
            ],
            "lattice_symmetries": [
                {
                    "id": "stable symmetry identifier",
                    "status": "exact|broken_by_boundary_or_discretization|continuum_only",
                    "operator_definition_id": "definition id for the lattice generator/operator",
                    "commutator_result": "exact finite-lattice result",
                    "target_eigenvalues": {
                        label: "exact eigenvalue fixed by that state definition, or unspecified"
                        for label in theory_contract["external_states"]["labels"]
                    },
                }
            ],
            "external_states": theory_contract["external_states"],
            "validation_obligations": [],
        }
        
        if full_paper_text:
            self.logger.info("=== 阶段 2A: 检测到提供了目标论文全文，跳过 arXiv 检索，直接从全文提取方案 ===")
            strategy_sys_instruction = "You are a precise data extraction assistant. Do not summarize or invent anything."
            strategy_prompt = f"""
Here is the text of a highly relevant physics paper about the {physics_target}:
{full_paper_text}

Historical complete-pipeline discretization candidates. A successful record is
evidence only for its original physical scope; reuse it only after checking the
current observable, external states, intermediate sectors and symmetries:
{discretization_experience or "No compatible historical discretization experience is available."}

Your task is to EXTRACT the exact mathematical definitions and rules proposed by the authors. DO NOT summarize. DO NOT change their notation.
Please locate and output the verbatim LaTeX expressions for the following:
1. How exactly are the gauge link operators (e.g., $U_n$) represented in their truncated space? (Extract the exact formula).
2. How exactly is the electric field (e.g., $E_n$ or $L_n$) represented? (Extract the exact formula).
3. If they explicitly wrote down the lattice Hamiltonian, extract that exact equation.
4. Any specific Hilbert space truncation limits (e.g., local dimension = 2, or spin-1/2 representation).

Output format: Just list the extracted formulas and the sentences immediately surrounding them. Be extremely strict.
"""
            strategy_advice = self._call_model_with_retry(strategy_prompt, strategy_sys_instruction)
            self.logger.info(f"调研完成，方案建议如下：\n{strategy_advice}\n")

        else:
            self.logger.info("=== 阶段 2A: 正在 arXiv 调研最前沿的离散化方案 ===")
            search_query = f'all:"{physics_target}" AND (all:"quantum simulation" OR all:"lattice gauge theory")'
            arxiv_abstracts = search_arxiv(search_query, max_results=3)

            strategy_sys_instruction = "You are a strategic physics consultant. Suggest methods based on papers."
            strategy_prompt = f"""
Here are some recent arXiv abstracts about the {physics_target}:
{arxiv_abstracts}

Historical complete-pipeline discretization candidates. A successful record is
evidence only for its original physical scope; reuse it only after checking the
current observable, external states, intermediate sectors and symmetries:
{discretization_experience or "No compatible historical discretization experience is available."}

Based on these, what is the best Fermion Discretization scheme and Gauge Field Digitization/Truncation scheme for a quantum computer?
CRITICAL RULE: Output ONLY the names of the methods and theoretical reasoning.
"""
            strategy_advice = self._call_model_with_retry(strategy_prompt, strategy_sys_instruction)
            self.logger.info(f"调研完成，方案建议如下：\n{strategy_advice}\n")


        self.logger.info("=== 阶段 2B: 根据调研方案，开始严密的公式推导 ===")
        
        derivation_sys_instruction = (
            "You are a world-class expert in Lattice Gauge Theory (LGT) and Quantum Computing. "
            "You MUST maintain strictly consistent notation with the user's continuous theory. "
            "Produce a direct finite-dimensional Hamiltonian whose coefficients depend only "
            "on declared parameters, never on an unknown target-state expectation value."
        )

        derivation_prompt = rf"""
You are a world-class expert in Lattice Gauge Theory. Your task is to apply a specific MATHEMATICAL METHOD to a specific TARGET PROBLEM.

==================================================
[TARGET PROBLEM - YOUR ULTIMATE GOAL]

Here is Agent 1's authoritative continuum-theory machine contract. It defines
the exact theory, observable and independent external states:
{tb}json
{agent1_report}
{tb}

Any entry in `resolved_conventions` is an already-fixed continuum definition,
not a discretization choice. Preserve its formula and numerical normalization
exactly in the lattice observable and expose any numerical constant needed by
execution in the Parameter Contract. Never replace it with a different gamma,
metric, light-cone, ordering, or operator-normalization convention.

Authoritative user-supplied simulation controls from the RunSpec (names and
values are immutable):
{tb}json
{json.dumps(simulation_parameters, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Controller execution request. This does not change the physics, but it states
whether the selected finite representation must support noiseless VQE and/or a
near-term hardware demonstration:
{tb}json
{json.dumps(run_spec.get("execution_request", {}), ensure_ascii=False, indent=2, sort_keys=True)}
{tb}
==================================================

==================================================
[HISTORICAL DISCRETIZATION EXPERIENCE]
This compact read-only evidence is ranked by complete-pipeline outcomes:
crosscheck_approved and analysis_complete outrank measurement-only results,
which outrank VQE-only convergence; crosscheck_not_approved is negative evidence.
{tb}json
{discretization_experience or "No compatible historical discretization experience is available."}
{tb}

DECISION RULE:
- First identify the symmetries required by the current observable and external
  states. Track the symmetry/charge sector after every ordered operator
  insertion, including sectors that occur only between insertions.
- Also read the Theory Contract's existing symmetry entries together with the
  observable coordinates. When supplied samples contain t and -t and the
  continuum theory at the supplied parameters has T, prefer a discretization
  that preserves T. When they contain spatial reflection pairs x and -x and
  the continuum theory has P, prefer one that preserves P. When both reverse,
  include an available PT symmetry. These are preservation priorities, not
  symmetries created by the grid: never impose one that the continuum theory
  lacks, never infer an unspecified external-state eigenvalue, and never infer
  an even/odd correlator identity without the required state and full-operator
  transformation law.
- When an external state specifies quantum numbers and a zero-based level index,
  preserve both exactly. Define the corresponding lattice quantum-number
  operators and sector; do not replace it by the unrestricted ground state.
- Treat every historical record as a candidate, not a mandatory answer. A
  successful record proves only that its bundle worked in its recorded scope.
  Reuse it when its boundary condition, fermion scheme, gauge treatment and
  truncation are compatible with the current symmetry and sector path, while
  re-deriving coefficients for current immutable parameters.
- Do not mix components from different historical records.
- If no candidate has established applicability, choose a new cohesive scheme.
  State the scientific compatibility reason for reuse or deviation in Strategy
  Alignment and in `experience_selection`.
- Historical values are evidence, not authority. RunSpec is authoritative for
  facts the user actually supplied, and the Theory Contract is authoritative
  for continuum equations and symmetries. A Theory-Contract statement that
  Agent1 did not infer a missing boundary condition, gauge fixing, regulator or
  truncation is stage-local, not a prohibition: choosing those missing
  discretization data is Agent2's responsibility.
==================================================

==================================================
[METHOD TOOLBOX - YOUR TOOLS]
Here is the reference-library material for the discretization scheme. Use it to
derive the selected method; a compatible reuse_recommended historical bundle
is useful prior evidence but does not override current physical requirements:
{tb}text
{full_paper_text}
{tb}

And here are formulas or strategic advice extracted from that material. Use
only the parts consistent with the selected cohesive bundle:
{strategy_advice}
==================================================

CRITICAL WARNINGS - READ CAREFULLY BEFORE WRITING:
1. SEPARATION OF CONCERNS: Use the [METHOD TOOLBOX] as literature evidence for lattice construction and gauge-field treatment, not as authority over the current theory. If it is absent or inapplicable, derive and justify a compatible construction independently; never attribute missing content to a source. 
2. DO NOT COPY RESULTS: DO NOT copy the final correlation function, vacuum states, or matrix elements from the reference paper. The paper solves a different physical problem.
3. STRICT EXTERNAL STATES: Preserve every named external state and every term in the requested matrix-element linear combination, including coefficients, bra labels, ket labels, and signs. In machine-readable JSON expressions, use the controller labels literally and never create renamed identifiers such as `hadron_lat` or `vacuum_lat`. The human LaTeX report may decorate the corresponding lattice states with a subscript such as $h_{{\rm lat}}$ without changing their machine identity. When no matrix_elements list is present, preserve the legacy $\langle\phi|O|\psi\rangle$ target.
4. ASYMMETRY WARNING: Bra and ket states in each term may differ. Do not use operator hermiticity or state identity unless the authoritative input permits it; derive every distinct matrix element before applying the requested linear combination.
5. DIRECT HAMILTONIAN: Discretize the unshifted continuum Hamiltonian directly.
Do not force its vacuum or ground-state energy to zero. Do not insert an unknown
target-state expectation value such as <vacuum|B|vacuum> into any Hamiltonian
coefficient or interaction operator. A later additive energy difference is
result analysis and is not part of H_lat.
6. SEPARATE OBSERVABLE SUBTRACTION: A requested linear combination such as
<h|O|h>-<vacuum|O|vacuum> already performs vacuum subtraction. Preserve its two
matrix elements and coefficient signs; do not hide the same subtraction inside
H_lat or redefine O using an unknown interacting-vacuum expectation value.
7. EXECUTABLE REGULATOR CHOICE: At finite lattice spacing, choose one explicit
finite operator prescription. If a normal-ordering reference is scientifically
necessary, it must be a fully specified fixed reference independent of the
unknown interacting target state. Prefer the direct finite-lattice composite
when it is well defined. Never create a self-consistent or circular Hamiltonian
unless the user explicitly requested and fully specified that different problem.

==================================================
CRITICAL HAMILTONIAN LOCALITY & BOUNDARY CONDITIONS TRADE-OFF:
Boundary conditions have no universal default. First retain only choices that
faithfully represent the requested states, observable, essential symmetries and
all intermediate sectors. Among those physically admissible choices, use
estimated near-term quantum-hardware cost as the main optimization criterion,
including qubit count, two-qubit depth, nonlocal couplings, routing/SWAP cost and
physical-sector enforcement cost.

- Priority 1 (Physics and symmetry): The scheme must represent the current
observable, external states and every intermediate symmetry/charge sector. It
should preserve exact continuum P, T or PT when that symmetry is relevant to
the supplied reflection-symmetric coordinate samples. If no physically
faithful candidate can preserve it, classify and explain the finite-lattice
breaking instead of silently claiming the symmetry.
- Priority 2 (Historical evidence): Prefer a crosscheck-approved bundle when
its recorded scope establishes applicability to those requirements.
- Priority 3 (Locality): Prefer local, nearest-neighbor interactions. OBC avoids
a wrap-around coupling, but exact Gauss-law elimination may produce a valid
nonlocal reduced Coulomb term and must be described honestly rather than
rejected automatically.
- Priority 4 (Physical Fidelity): Consider whether the observable severely
suffers from boundary/finite-size effects or requires translational invariance.

INSTRUCTION: You must explicitly weigh these priorities mathematically.
1. Reject any boundary choice that changes the requested physical target.
2. Among the remaining valid choices, select the lowest estimated near-term
   circuit cost; OBC or PBC may win depending on the model and observable.
3. If every physically faithful choice exceeds plausible near-term resources,
   retain the scientifically correct choice and mark it as simulator/reference
   only rather than silently changing the physics.
You MUST include a brief "Locality vs. Physics Trade-off" justification in your mathematical report explaining your choice of boundary conditions based on this priority hierarchy.
==================================================

==================================================
GAUGE-REPRESENTATION FEASIBILITY RULE:
For every gauge theory, do not choose `gauge_fixing` or
`gauss_law_elimination` in isolation. First compare all physically admissible
representations among: the complete link Hilbert space, exact physical-degree-
of-freedom/Gauss-law elimination, and a partial or hybrid reduction. Include
loop, plaquette, dual, quantum-link, or finite-group variables only when they
are valid for the stated theory, boundary conditions, global sectors, states,
and observable.

Apply the following decision order:
1. PHYSICS GATE: Reject a representation that changes the target theory,
   removes a required boundary/topological sector, makes an external state or
   observable undefined, or violates an essential exact symmetry.
2. EXECUTABILITY GATE: For every remaining representation, calculate or bound
   the kinematic Hilbert-space dimension, physical-sector dimension, number of
   independent constraints, and corresponding qubit counts. Estimate whether
   the requested noiseless test can materialize the required operator and
   sector representation. In particular, report the memory scaling of any
   proposed dense Hamiltonian or explicit embedded physical basis,
   `16*dim(H_kin)^2` bytes and
   `16*dim(H_kin)*dim(H_phys)` bytes respectively for complex128 data. Never
   select a representation that requires such an evidently infeasible object
   when a physically equivalent executable reduction exists.
3. OPERATOR-COST COMPARISON: Compare the resulting Hamiltonian term growth,
   maximum operator support/locality, induced long-range interactions,
   observable-string support, physical-sector enforcement cost, and expected
   routing/two-qubit-gate burden. Fewer qubits alone is not sufficient.
4. SELECTION: Choose the physically valid executable representation with the
   lowest combined cost. Use full link variables when their locality advantage
   outweighs a feasible constraint cost; use exact reduction when the full
   kinematic space is infeasible and reduced operators remain manageable; use
   partial reduction when the two extremes are both costly.

If no physically faithful representation is executable under the supplied
request, say so explicitly instead of claiming executability or silently
changing the physics. Agent2 owns this physical representation choice. Agent3
will translate it literally and must not reverse gauge fixing, truncation, or
Gauss-law elimination.

Record the selected representation's kinematic dimension, physical dimension,
independent-constraint count, effective degree/qubit-count estimate, and
locality trade-off in the existing gauge `discretizations[].settings` and
`field_encodings[].degrees_of_freedom` descriptions; do not invent a new
contract type or runtime algorithm.

You MUST include a concise subsection named
`\subsection*{{Gauge Representation Trade-off}}` comparing the admissible full,
reduced, and partial/hybrid choices and explaining why the selected one passes
both the physics and executability gates.
==================================================

==================================================
SINGLE-SOURCE LATTICE-GEOMETRY RULE:
Treat the immutable RunSpec lattice shape as the sole user authority. Record
the selected boundary condition, shape/index ranges, and directly derived
vertex, link, plaquette/cell and independent-constraint counts exactly once in
the spacetime `discretizations[].settings`. Do not create separate runtime
parameters for counts that are algebraically fixed by that shape and boundary
condition.

Other entries must reference that selected geometry instead of copying its
literal counts into prose. In particular, phrase physical-sector obligations
as applying at `every lattice vertex` rather than repeating a vertex number,
and let field-encoding descriptions reference the authoritative spacetime and
gauge settings. Keep a representation-specific effective degree/qubit count in
the gauge settings only when downstream execution needs it. This rule removes
duplicate bookkeeping; it does not relax the requirement that all formulas and
index ranges agree with the authoritative geometry.
==================================================

==================================================
WILSON SECOND-DERIVATIVE RULE (ALL SPATIAL DIMENSIONS):
When Wilson fermions are selected, discretize the Wilson regulator directly as
the gauge-covariant spatial second derivative (lattice Laplacian), separately
in every spatial direction. This applies in 1+1, 2+1, 3+1, and higher
dimensions; do not specialize the reasoning to a one-dimensional chain. For
isotropic spacing a and d spatial dimensions, the open-lattice Hamiltonian
form has the full onsite Wilson coefficient r*d/a at every retained site and
the corresponding -r/(2*a) gauge-covariant hopping on every retained nearest-
neighbor link in each direction (with the appropriate Dirac beta/gamma
matrix). For anisotropic lattices, use the direction-dependent sum r_i/a_i.

Do not replace this direct second-difference operator by a sum of products of
first differences, such as a linkwise gradient-squared form, under open or
otherwise bounded boundary conditions unless every discrete summation-by-parts
boundary term is retained and the resulting finite-lattice operator is shown
algebraically identical. In particular, an endpoint, edge, or corner onsite
coefficient must not be silently reduced merely because that site has fewer
retained links. Under periodic boundaries the two representations may be
treated as equivalent after an explicit algebraic check. This rule fixes the
finite-volume boundary prescription; it does not force Wilson fermions when a
different fermion discretization is scientifically preferable.
==================================================

Your task is to choose and derive a finite-dimensional lattice Hamiltonian. Use
the strategic advice and historical records as candidates, but make the final
choice from the current observable, external states, required symmetries and
intermediate-sector path.
Please systematically execute the following steps:

a) Gauge fixing conditions: Based on the observable, external states, boundary
conditions and global sectors, choose an applicable gauge treatment (including
no gauge fixing). In 1+1 dimensions, exact link elimination is often compatible
with axial gauge under OBC; temporal gauge alone generally does not eliminate
spatial links, while PBC may retain a global holonomy. Derive the applicable
case instead of imposing one gauge choice universally.
b) Boundary Conditions: Select the appropriate boundary conditions (OBC vs. PBC) by strictly following the Locality vs. Physics Trade-off instructions above. Include the required explicit justification.
c) Fermion Discretization: When fermions are present, evaluate applicable
fermion discretizations and select one from the symmetries and observable. When
they are absent, state that this step is not applicable and invent no fermion
sector. If Wilson fermions are selected, apply the dimension-independent
Wilson Second-Derivative Rule above in every spatial direction.
d) Gauge Field Discretization: When gauge fields are present, compare the
formulations applicable to the current theory, which may include a
Kogut--Susskind link formulation, a quantum-link/loop formulation, a finite-group
exact encoding, exact Gauss-law elimination, or a partial/hybrid reduction.
Apply the Gauge-Representation Feasibility Rule above rather than automatically
preferring either complete links or complete elimination. Preserve an upstream
lattice Hamiltonian formulation when it already defines the target, but still
distinguish its physical formulation from a computationally infeasible
redundant representation. After exact elimination, do not reintroduce an
independent link Hilbert space.
e) Gauge Field Digitization: If the selected local gauge Hilbert space is
infinite-dimensional, compare controlled truncation schemes. If it is already
finite-dimensional, preserve its exact representation without calling it an
approximate truncation. Select the most suitable representation for the current
physics and hardware constraints and explain why.
f) Derivation: Derive the explicit expressions for the Hamiltonian and the observable in this finite-dimensional lattice Hilbert space. If the observable contains Wilson line, write down how the Wilson line looks like on lattice after choosing the gauge conditions in step a) and treat the Wilson line carefully.
g) Lattice Symmetries: Identify the residual symmetries surviving on the lattice and provide the explicit lattice operator expressions for the conserved quantities ($Q_{{lat}}$).
h.0) Symmetry Classification: In `lattice_symmetries`, classify every symmetry
relevant to the Hamiltonian, target states, or observable as `exact`,
`broken_by_boundary_or_discretization`, or `continuum_only` for the selected
finite lattice and boundary condition. Link each symmetry to its explicit
lattice operator definition and state the actual finite-lattice commutator or
transformation result. For each named external state, record a target
eigenvalue only when its authoritative state definition fixes that quantum
number; otherwise write `unspecified`. Charge zero alone must not be treated as
a specified charge-conjugation parity, and degeneracy must not be silently
resolved by an arbitrary symmetry eigenvalue.
h) Commutator Check: Explicitly calculate the commutator $[H_{{lat}}, Q_{{lat}}]$ for the chosen finite-dimensional truncation. Report the result truthfully: do NOT assume or force it to vanish. If it is nonzero, identify the truncation or boundary terms responsible instead of fabricating a zero proof.
i) Operator-sector path: In the stated operator ordering, list the symmetry or
charge sector after each insertion. Verify that the Hamiltonian, boundary-flux
relations, Gauss-law reconstruction and observable are defined consistently in
every visited sector. Do not project the evolution onto only the external-state
sector when an insertion creates a different intermediate sector.

Keep the human report concise. It is an audit view, not a machine interface.

OUTPUT FORMAT REQUIREMENTS:
Output a complete, compilable LaTeX document containing:
(1) Choices and justifications for gauge fixing conditions.
(2) Choices and justifications for Boundary conditions (must include the "Locality vs. Physics Trade-off"), Fermion discretization, and Gauge digitization schemes.
(2a) The required "Gauge Representation Trade-off", including Hilbert-space
dimensions, independent constraints, effective qubit counts, locality changes,
and the executability conclusion for the selected representation.
(3) The specific expression of the unshifted finite-dimensional Hilbert-space
Hamiltonian on the lattice, with no unknown state-dependent coefficients.
(4) The specific expressions of all conserved quantities on the lattice.
(5) The step-by-step commutator calculation and its actual result for the chosen finite-dimensional truncation.
(6) The specific expression of the observable $O$ in the finite-dimensional LGT Hilbert space.
(7) The final exact lattice expression for the complete requested matrix-element linear combination.

Inside the document, before doing the math, you MUST include a subsection named "\subsection*{{Strategy Alignment}}" explaining which strategic suggestions apply and why any incompatible suggestion is rejected in favor of authoritative physical inputs.
Also include a subsection named "\subsection*{{Simulation Parameter Contract}}" summarizing every model-specific simulation parameter, its chosen value or unresolved status, and all compatibility constraints.

After the LaTeX block, output one authoritative JSON parameter contract. Parameter names are NOT fixed: declare only the parameters relevant to this model and discretization. Set `model.gauge_structure` to the actual local gauge group for a gauge theory and to `none` for a non-gauge theory; downstream Gauss-sector requirements depend on this distinction. In the extensible `discretizations` list, separately identify every applicable sector (for example spacetime, fermion, boson, and gauge), its named scheme and model-specific settings. `parameter_references` is only an optional lookup index: include only exact names already declared in `parameters`, and omit an uncertain alias rather than declaring a duplicate parameter. Every gauge-sector component must explicitly describe its digitization or elimination and its truncation settings; use scheme `none` when gauge variables are exactly eliminated or genuinely untruncated. Do not add inapplicable sector placeholders. Include compatibility rules that genuinely affect truncation, boundary conditions, charge/Gauss-law sectors, lattice geometry, or observable ranges. For each required run parameter, preserve a user-supplied value or derive its direct consequence. Choose a missing discretization control only when it belongs to this stage and is scientifically justified. If a required physical value cannot be resolved, stop with a concrete explanation; do not emit an executable contract with a null required value. Add runtime validation obligations wherever consistency cannot be guaranteed from scalar ranges alone, including a nonempty physical-sector check when applicable. State only the physical condition that must hold; do NOT recommend a qubit-level or software verification algorithm, because Agent 3 owns that mapping. The JSON is authoritative; the LaTeX report is a concise audit and need not duplicate metadata field by field.

The Parameter Contract must not duplicate fields already owned by the
Observable Specification. In particular, do not add `requested_components`,
observable names/expressions, insertion lists, matrix-element lists, ordering,
or sampling grids as parameters. They remain available downstream through the
separate Observable Specification.

An executable final contract may not contain `required_for_run=true` together
with `selected_value=null`. If a quantity is fixed by the observable coordinate
convention, derive it. For `lattice_indices`, physical spatial and temporal
coordinates equal the supplied index times the selected lattice spacing in
natural units c=1; for example t_index=1 and a=1 imply physical time 1. If a
safe discretization control is genuinely yours to choose, select and justify
it. Mathematical path pullbacks, operator definitions, ordering rules and
algorithmic prescriptions are definitions or validation obligations, NOT
independent runtime parameters. Never create a required string parameter merely
to hold such a prescription.

Agent2 owns the exact lattice definition, not its later quantum-simulation
algorithm. A statement that this stage does not prescribe a numerical time
step, quadrature, product formula, or Trotter method MUST NOT be converted into
a fatal constraint forbidding Agent4 from choosing a controlled convergent
implementation of the same exact operator. Preserve exact path geometry,
orientation, ordering, and the continuum/large-step target; leave runtime
approximation choice explicitly to Agent4.

Every simulation/discretization control explicitly supplied in the Agent 1
report is immutable: preserve its exact name and value as a selected contract
parameter. Resolve every directly determined dependent parameter as well (for
example, if the selected discretization implies `fermionic_mode_count = 2 *
spatial_site_count`, calculate it). Derived lattice counts are the exception to
repeating them as parameters: calculate them once in the authoritative
spacetime discretization settings under the Single-Source Lattice-Geometry
Rule. Do not leave a required parameter null when the user controls and
declared compatibility relations determine it exactly.

In the JSON `external_states` object, preserve all controller-supplied state labels and definitions in their original order. Put labels in the same `equivalent_groups` entry ONLY when Agent 1 explicitly states that the corresponding physical states are identical; otherwise leave them independent. Equality of quantum numbers alone is not sufficient to infer identical states.

{PARAMETER_CONTRACT_SCHEMA_GUIDANCE}

After those two blocks, return one additional fenced JSON lattice-discretization
contract using the envelope below. Replace every placeholder with exact indexed
definitions.
Do not copy Parameter Contract external_states or validation_obligations into
the lattice contract; downstream stages read those facts from their single
authoritative source. The two JSON contracts
are authoritative; the concise LaTeX is only a human audit and must not embed
JSON, quantikz, tikz, figures or verbatim.
{tb}json
{json.dumps(discretization_example, ensure_ascii=False, indent=2)}
{tb}

Return exactly one {tb}latex block followed by the parameter-contract JSON and
the lattice-discretization JSON. No conversational text outside those blocks.
"""

        final_content = self._call_model_with_retry(derivation_prompt, derivation_sys_instruction)
        if raw_response_path:
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(final_content)
                if final_content and not final_content.endswith("\n"):
                    handle.write("\n")
        
        self.logger.info("Agent 2 离散化方案推导完成！")
        lattice_report, parameter_contract, discretization_contract = (
            extract_discretization_response(final_content, theory_contract, run_spec)
        )
        return lattice_report, strategy_advice, parameter_contract, discretization_contract

    def revise_from_review(
        self,
        agent1_report,
        theory_contract,
        run_spec,
        parameter_contract,
        discretization_contract,
        findings,
        raw_response_path=None,
        previous_report=None,
    ):
        """Apply one bounded correction to the identified Agent2 entries."""
        tb = chr(96) * 3
        prompt = rf"""
Correct only the erroneous Agent2 entries and their scientifically necessary
dependent entries identified below. Do not regenerate either complete contract.

Authoritative continuous-theory reference:
{tb}json
{agent1_report}
{tb}

Immutable controller RunSpec:
{tb}json
{json.dumps(run_spec, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Previously validated Parameter Contract:
{tb}json
{format_parameter_contract(parameter_contract)}
{tb}

Previously validated lattice-discretization contract:
{tb}json
{format_machine_contract(discretization_contract, validate_discretization_contract, expected_theory_contract=theory_contract, expected_parameter_contract=parameter_contract)}
{tb}

Fatal review findings to correct:
{tb}json
{json.dumps(findings, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Return exactly one JSON object containing `parameter_contract` and/or
`discretization_contract`, whichever is affected. Inside them include only corrected complete entries:
identify parameters by `name`, constraints and validation obligations by `id`,
discretizations and field encodings by `sector`, definitions and lattice
symmetries by `id`, and include only corrected `model` or `lattice` keys.
Omit every unchanged entry and section. This is a stage-specific correction
envelope, not a general JSON Patch; controller code preserves omitted content.
Correct every fatal finding while preserving all valid scientific choices and
all immutable user inputs.
For a geometry correction, update the one authoritative spacetime geometry
entry and every affected equation or index range. Remove stale copied numbers
from descriptive fields by referring to `every lattice vertex` or the selected
geometry; do not propagate the same derived count through multiple prose
fields.
Use every controller-supplied state label literally in machine-readable JSON
expressions. A lattice-state decoration may appear only in the human LaTeX
report; it must not create identifiers such as `hadron_lat` or `vacuum_lat`.
Preserve the finite-lattice classification, operator link, commutator result,
and non-inferred target eigenvalues of every `lattice_symmetries` entry unless
a reviewer finding identifies a concrete scientific error in that entry.

Agent2 retains responsibility for boundary conditions, fermion
discretization, gauge fixing, and gauge truncation or exact elimination when
the user did not specify them. Never retreat to an unresolved Hamiltonian
family merely because Agent1 correctly left a discretization choice open.
If a finding identifies an infeasible gauge representation, revise the selected
full, reduced, or partial/hybrid treatment together with every necessarily
dependent Hamiltonian, observable, field-encoding, symmetry, constraint, and
dimension statement. Do not patch only the resource prose, and do not defer the
physical representation choice to Agent3.

Keep any corrected Hamiltonian entry direct and unshifted. Never make a coefficient
depend on an unknown target-state or interacting-vacuum expectation value. Keep
vacuum subtraction expressed by the requested matrix-element linear combination;
do not embed it in H_lat or redundantly inside O. Any necessary normal-ordering
reference must be fixed explicitly and independently of the state being solved.
When correcting a Wilson-fermion Hamiltonian on a bounded lattice in any
spatial dimension, preserve the direct gauge-covariant second-difference
(lattice-Laplacian) prescription in every direction. Do not substitute a
linkwise product of first differences unless all summation-by-parts boundary
terms are retained and the finite operator is algebraically identical; in
particular, do not silently halve onsite Wilson coefficients at endpoints,
edges, or corners.

The merged result must be executable: every `required_for_run=true` parameter
must have a non-null selected value. Apply the RunSpec coordinate convention:
under `lattice_indices`, multiply spatial and temporal indices by the selected
lattice spacing in natural units c=1. Do not represent a path pullback,
operator formula, ordering rule, or algorithmic prescription as an independent
runtime parameter; encode it in the discretization definitions or validation
obligations instead.
"""
        content = self._call_model_with_retry(
            prompt,
            "You are the original Agent2 lattice-discretization physicist. "
            "Correct only the identified entries using reviewer findings while "
            "preserving controller-owned inputs.",
        )
        if raw_response_path:
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(content)
                if content and not content.endswith("\n"):
                    handle.write("\n")
        corrected_parameter, corrected_discretization = apply_agent2_correction(
            content,
            parameter_contract,
            discretization_contract,
            theory_contract,
            run_spec,
        )
        report = previous_report or (
            "\\documentclass{article}\n\\begin{document}\n"
            "Agent2 local scientific correction applied.\n\\end{document}\n"
        )
        return report, corrected_parameter, corrected_discretization


# ==========================================
# Agent 2.5: 格点理论审查智能体 (Reviewer)
# ==========================================
class LatticeReviewerAgent:
    def __init__(self, api_key, base_url=None, model_name="gpt-5.6-sol"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.logger = setup_logger("Agent2.5_LatticeReviewer")
        
        self.system_instruction = (
            "You are a strict, senior reviewer specializing in LGT quantum simulation. "
            "Your duty is to scrutinize lattice Hamiltonian derivations, gauge link truncations, "
            "observable discretizations, parameter substitutions, and commutator algebra to "
            "prevent scientific errors. Review physics only: never judge workflow authority, "
            "historical-policy compliance, provenance, formatting, schema style, or whether "
            "Agent2 was permitted to choose a discretization."
        )

    def review(
        self, agent1_report, theory_contract, parameter_contract,
        discretization_contract, run_spec, raw_response_path, review_status_path,
        discretization_experience=None,
    ):
        # Reviewers receive no historical-selection or workflow-authorization
        # task. Historical evidence remains outside their scientific verdict.
        del discretization_experience
        tb = chr(96) * 3
        deterministic_findings = required_run_parameter_findings(parameter_contract)
        prompt = rf"""
Review two already schema-valid Agent2 machine contracts. Do not regenerate
either contract and do not return LaTeX. Return only one compact JSON review.

Continuous-theory reference:
{tb}json
{agent1_report}
{tb}

Validated parameter contract:
{tb}json
{format_parameter_contract(parameter_contract)}
{tb}

Validated lattice-discretization contract:
{tb}json
{format_machine_contract(discretization_contract, validate_discretization_contract, expected_theory_contract=theory_contract, expected_parameter_contract=parameter_contract)}
{tb}

Controller execution request used only to assess whether the selected finite
representation is executable without changing the target physics:
{tb}json
{json.dumps(run_spec.get("execution_request", {}), ensure_ascii=False, indent=2, sort_keys=True)}
{tb}

Deterministic executable-interface findings already known to the controller:
{tb}json
{json.dumps(deterministic_findings, ensure_ascii=False, indent=2, sort_keys=True)}
{tb}
These findings are part of the same review. Continue through the entire
scientific checklist; do not stop after confirming one of them.

Please strictly verify:
1. When a gauge sector is present, is its selected exact encoding, elimination,
   or truncation valid and finite-dimensional for quantum computation? For a
   non-gauge theory, treat this check as not applicable.
2. Does the selected lattice formulation reconstruct the intended Hamiltonian
   with correct operators and coefficients? Check Kogut--Susskind electric/link/
   plaquette terms only when that is the selected formulation.
3. EXTREMELY IMPORTANT: Recalculate $[H_{{lat}}, Q_{{lat}}]$ step-by-step without presupposing the answer. If the finite truncation gives a nonzero result, preserve and explain it; never alter valid operators merely to force the commutator to zero.
4. Does the lattice observable expression correctly incorporate the Wilson line (gauge link matrices) to maintain lattice gauge invariance?
5. Do the Hamiltonian and observable use the supplied numerical parameters and
   their direct algebraic consequences correctly? Check domains and physical
   compatibility conditions only where they affect the equations, Hilbert
   space, boundary terms, charge sector, or physical-sector nonemptiness.
6. Are all named states and every coefficient, bra label, ket label, and sign in
   the matrix-element combination physically preserved without silent
   identification? Structured state identity comes only from the validated
   Parameter Contract. Do not treat an unambiguous human formula decoration
   such as $h_{{\rm lat}}$, or its textual spelling `hadron_lat`, as a new state
   when it plainly denotes the lattice representation of the existing
   `hadron` label. Such notation is nonblocking; report a fatal finding only
   when a genuinely different or ambiguous physical state is introduced.
7. Following the observable's stated operator ordering, are the Hamiltonian,
   Gauss-law reconstruction and boundary conditions valid in every intermediate
   symmetry/charge sector reached after an insertion, rather than only in the
   external-state sector?
8. Is H_lat a direct, unshifted and executable operator whose coefficients use
   only declared parameters? Reject as a fatal scientific error any zero-ground-
   energy requirement or any coefficient depending on an unknown target-state
   expectation value. Verify that requested vacuum subtraction remains an
   explicit matrix-element linear combination rather than a circular definition
   of H_lat or a redundant state-dependent redefinition of O.
9. Does `lattice_symmetries` correctly distinguish exact finite-lattice
   symmetries from boundary/discretization-broken and continuum-only ones, link
   each to the right lattice operator, and avoid assigning an external-state
   eigenvalue that its authoritative definition does not fix?
   For supplied t <-> -t or x <-> -x sample pairs, verify that every
   corresponding continuum T, P or PT symmetry identified by Agent1 was
   considered in the boundary/discretization choice and either preserved or
   truthfully classified as broken. A symmetric grid is not by itself evidence
   for a symmetry or for an even/odd correlator identity.
10. Gauge-representation selection: Did Agent2 compare every physically
    admissible complete-link, exact reduced, and partial/hybrid representation?
    Verify the kinematic and physical Hilbert-space dimensions, independent
    constraint count, effective qubit estimates, and the stated locality/term-
    growth/observable-support trade-off. Reject as fatal a representation
    claimed to be executable when its required dense operator or explicit
    embedded sector basis is evidently infeasible and a physically equivalent
    executable representation exists. Do not demand Gauss-law elimination when
    it removes a required sector or when its induced nonlocal cost is worse than
    a feasible full or partial encoding. Agent3 may estimate mapping resources,
    but it must not be expected to change Agent2's physical representation.
11. If Wilson fermions were selected, was the regulator constructed directly
    from the gauge-covariant lattice second derivative in every spatial
    direction? Under bounded boundaries, expand the operator and verify the
    full prescribed onsite Wilson coefficient at endpoints, edges, and corners
    as well as the link hopping coefficients. Reject as fatal an unannounced
    replacement by a first-difference-product form that drops the discrete
    summation-by-parts boundary term. Apply this check dimension-independently;
    for periodic directions, accept an algebraically equivalent representation.

For geometry consistency, read the shape and boundary condition from the
authoritative spacetime discretization settings and independently recompute the
necessary counts. A wrong equation, index range, physical constraint, or
effective Hilbert-space dimension is fatal. A stale number occurring only in
non-executable explanatory prose is a warning and must not by itself block the
pipeline.

Agent2 owns the choice of lattice boundary condition, fermion discretization,
gauge fixing, gauge truncation or exact gauge elimination when the user did not
fix them. Treat those selected choices as the premise of the review. A
continuum-theory statement that Agent1 did not choose such data is not a
prohibition on Agent2. Review whether the resulting choice and formulas are
scientifically valid for the stated model, states and observable; do not review
who authorized the choice, why it was selected, or whether it follows a
historical record. For a continuum path without an elementary lattice link,
assess the proposed lattice path by endpoints, orientation, gauge covariance
and continuum limit rather than demanding a nonexistent diagonal link.

This is a read-only review. Never rewrite or patch either contract. Use verdict
`needs_revision` only for a concrete scientific error in an equation,
coefficient, operator, state, parameter substitution, physical constraint, or
continuum-limit construction; classify that finding as `fatal`. Use `warning`
only for a nonblocking scientific limitation. Do not report formatting,
metadata, provenance, authorization, ownership, or historical-policy findings.
An overall phase/sign convention or alternative symmetry-generator label is a
warning when it leaves every implemented projector, selected sector, ansatz
restriction, observable and reported physical comparison unchanged. It is
fatal only when the inconsistent eigenvalue or sector label is actually used
downstream or changes a requested physical conclusion.
For every fatal finding, name the affected contract and its existing natural
identifier (`parameter.name`, `constraint.id`, `definition.id`,
`lattice_symmetry.id`, or `field_encoding.sector`) and name only the dependent
entries that must change with it.

Finish all eleven checks before returning. Report every independently discovered
scientific error in this one response, even when one fatal error is already
known. Do not return early after the first failure.

Return one compact JSON object and no other text. The controller owns and
injects schema/version/type metadata; provide only the semantic decision:
{tb}json
{{
  "verdict": "accept",
  "findings": []
}}
{tb}
"""
        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": prompt}
        ]

        self.logger.info("正在将两个机器契约送交只读 Reviewer（单次 review）...")
        start_time = time.time()
        full_content = ""
        model_calls = 1
        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name, messages=messages, stream=True
            )
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
            with open(raw_response_path, "w", encoding="utf-8") as handle:
                handle.write(full_content)
                if full_content and not full_content.endswith("\n"):
                    handle.write("\n")
            decision = extract_review_decision(full_content, "agent2_review")
            combined_findings = list(deterministic_findings)
            known_messages = {
                item.get("message") for item in combined_findings
                if isinstance(item, dict)
            }
            for finding in decision["findings"]:
                if finding.get("message") not in known_messages:
                    combined_findings.append(finding)
                    known_messages.add(finding.get("message"))
            decision["findings"] = combined_findings
            if deterministic_findings:
                decision["verdict"] = "needs_revision"
            reviewed_parameter = validate_parameter_contract(parameter_contract)
            reviewed_discretization = validate_discretization_contract(
                discretization_contract,
                expected_theory_contract=theory_contract,
                expected_parameter_contract=reviewed_parameter,
            )
            blocked = decision["verdict"] == "needs_revision"
            record = {
                "review_status_schema_version": "1.0",
                "stage": "agent2",
                "status": "accepted" if not blocked else "needs_revision",
                "verdict": decision["verdict"],
                "findings": decision["findings"],
                "model_calls": model_calls,
                "raw_response_path": os.path.abspath(raw_response_path),
                "machine_chain_blocked": blocked,
            }
            atomic_json_write(review_status_path, record)
            if blocked:
                self.logger.error(
                    "Reviewer 发现 fatal 科学问题；已验证 draft 保持不变。"
                )
                return (
                    render_review_audit(decision), reviewed_parameter,
                    reviewed_discretization, record,
                )
            self.logger.info("Reviewer 只读审查完成，耗时: %.2f 秒", time.time() - start_time)
            return (
                render_review_audit(decision), reviewed_parameter,
                reviewed_discretization, record,
            )
        except Exception as error:
            if not os.path.isfile(raw_response_path):
                with open(raw_response_path, "w", encoding="utf-8") as handle:
                    handle.write(full_content)
            if deterministic_findings:
                decision = {
                    "verdict": "needs_revision",
                    "findings": deterministic_findings,
                }
                record = {
                    "review_status_schema_version": "1.0",
                    "stage": "agent2",
                    "status": "needs_revision",
                    "verdict": "needs_revision",
                    "findings": deterministic_findings,
                    "error": "{}: {}".format(type(error).__name__, error),
                    "model_calls": model_calls,
                    "raw_response_path": os.path.abspath(raw_response_path),
                    "machine_chain_blocked": True,
                }
                atomic_json_write(review_status_path, record)
                return (
                    render_review_audit(decision), parameter_contract,
                    discretization_contract, record,
                )
            record = {
                "review_status_schema_version": "1.0",
                "stage": "agent2",
                "status": "review_inconclusive_nonblocking",
                "error": "{}: {}".format(type(error).__name__, error),
                "model_calls": model_calls,
                "raw_response_path": os.path.abspath(raw_response_path),
                "machine_chain_blocked": False,
            }
            atomic_json_write(review_status_path, record)
            self.logger.warning(
                "Reviewer 输出无法结构化；保留已验证 draft 并继续: %s", error
            )
            fallback = {
                "verdict": "accept",
                "findings": [{
                    "severity": "warning",
                    "message": "Reviewer response could not be parsed; raw response was retained.",
                }],
            }
            return (
                render_review_audit(fallback), parameter_contract,
                discretization_contract, record,
            )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Agent2 discretization and review.")
    parser.add_argument("--theory-contract", required=True, help="Authoritative Agent-1 machine contract.")
    parser.add_argument("--run-spec", required=True, help="Immutable controller-owned RunSpec.")
    parser.add_argument("--library", default="A2_library")
    parser.add_argument(
        "--experience-db",
        help="Read-only simulation experience database used to rank prior discretizations.",
    )
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--output-manifest")
    parser.add_argument("--output-tag")
    parser.add_argument("--output-root")
    return parser.parse_args()


# ==========================================
# 独立测试工作流
# ==========================================
if __name__ == "__main__":
    args = parse_arguments()
    output_tag = resolve_output_tag(args.output_tag)
    report_directory = categorized_output_directory(args.output_root, "reports")
    contract_directory = categorized_output_directory(args.output_root, "contracts")
    audit_directory = categorized_output_directory(args.output_root, "audit")
    YOUR_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not YOUR_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY before running this agent.")
    run_spec = load_run_spec(args.run_spec)
    states = run_spec["user_input"]["external_states"]
    theory_contract = load_machine_contract(
        args.theory_contract,
        validate_theory_contract,
        expected_model=run_spec["user_input"]["model"],
        expected_observable=run_spec["user_input"]["observable"],
        expected_external_states=states,
    )
    real_agent1_report = format_machine_contract(theory_contract, validate_theory_contract)
    physics_task = run_spec["user_input"]["model"]
    print("从 RunSpec 继承物理模型: {}".format(physics_task))
    current_problem = summarize_discretization_problem(
        run_spec["user_input"]["observable"],
        states,
        gauge_structure=theory_contract.get("model", {}).get("gauge_structure"),
        theory_definitions=theory_contract.get("definitions", []),
    )
    discretization_experience = format_discretization_experience_context(
        args.experience_db,
        physics_task,
        run_spec["user_input"]["parameters"],
        gauge_structure=theory_contract.get("model", {}).get("gauge_structure"),
        current_problem=current_problem,
    ) if args.experience_db else (
        "No compatible historical discretization experience is available."
    )
    if args.experience_db:
        print("Agent2 已读取并排序历史离散化经验: {}".format(args.experience_db))
    theory_text = (physics_task + " " + real_agent1_report).lower()
    keywords_for_agent2 = ["discretization", "Hamiltonian", physics_task]
    if theory_contract.get("model", {}).get("gauge_structure", "none").lower() != "none":
        keywords_for_agent2 += ["gauge field", "link operator"]
    if "ferm" in theory_text or "quark" in theory_text:
        keywords_for_agent2 += ["fermion"]
    if "kogut" in theory_text or "susskind" in theory_text:
        keywords_for_agent2 += ["Kogut-Susskind", "electric field", "plaquette"]
    if "truncat" in theory_text or "u(1)" in theory_text or "su(" in theory_text:
        keywords_for_agent2 += ["truncation"]

    (
        draft_file,
        draft_parameter_file,
        draft_discretization_file,
    ) = reusable_or_new_draft_paths(
        report_directory, contract_directory, output_tag
    )
    generator_raw_file = next_generator_raw_path(audit_directory, output_tag)

    # A validated draft is a paid-stage checkpoint. On resume, reuse it and
    # invoke only the compact reviewer instead of repeating the derivation.
    agent2 = LatticeDiscretizationAgent(
        api_key=YOUR_API_KEY,
        base_url=args.base_url,
        model_name=args.model
    )
    if all(os.path.isfile(path) for path in (
        draft_file, draft_parameter_file, draft_discretization_file
    )):
        with open(draft_file, "r", encoding="utf-8") as handle:
            lattice_draft_latex = handle.read()
        draft_parameter_contract = load_parameter_contract(draft_parameter_file)
        draft_discretization_contract = load_machine_contract(
            draft_discretization_file,
            validate_discretization_contract,
            expected_theory_contract=theory_contract,
            expected_parameter_contract=draft_parameter_contract,
        )
        agent2.logger.info("读取已验证 Agent2 初稿 checkpoint；跳过付费推导。")
    else:
        replayed_raw = latest_generator_raw_path(audit_directory, output_tag)
        recovered = None
        if replayed_raw:
            try:
                with open(replayed_raw, "r", encoding="utf-8") as handle:
                    recovered = extract_discretization_response(
                        handle.read(), theory_contract, run_spec
                    )
                agent2.logger.info(
                    "零成本重新验证已保存的 Agent2 generator 原始响应: %s",
                    replayed_raw,
                )
            except Exception as error:
                agent2.logger.warning(
                    "已保存的 Agent2 generator 响应仍无法验证，将进行一次新调用: %s",
                    error,
                )
        if recovered is not None:
            (
                lattice_draft_latex,
                draft_parameter_contract,
                draft_discretization_contract,
            ) = recovered
        else:
            my_paper_text = extract_relevant_sections_from_library(
                library_path=args.library,
                target_keywords=keywords_for_agent2,
                max_chars=15000,
                primary_keywords=[physics_task],
            )
            (
                lattice_draft_latex,
                _strategy_advice,
                draft_parameter_contract,
                draft_discretization_contract,
            ) = agent2.process_continuous_theory(
                real_agent1_report,
                theory_contract,
                run_spec,
                physics_target=physics_task,
                simulation_parameters=run_spec["user_input"].get(
                    "parameter_groups", {}
                ).get("simulation", {}),
                full_paper_text=my_paper_text,
                raw_response_path=generator_raw_file,
                discretization_experience=discretization_experience,
            )
        with open(draft_file, "w", encoding="utf-8") as handle:
            handle.write(lattice_draft_latex)
        save_parameter_contract(draft_parameter_contract, draft_parameter_file)
        save_machine_contract(
            draft_discretization_contract,
            draft_discretization_file,
            validate_discretization_contract,
            expected_theory_contract=theory_contract,
            expected_parameter_contract=draft_parameter_contract,
        )
        agent2.logger.info("Agent2 已验证初稿和机器契约 checkpoint 已保存。")

    # 2. Run Agent 2.5. At most two bounded local corrections are allowed;
    # complete-contract regeneration is never used here.
    reviewer = LatticeReviewerAgent(
        api_key=YOUR_API_KEY,
        base_url=args.base_url,
        model_name=args.model
    )
    reviewer_raw_file, reviewer_status_file = next_reviewer_attempt_paths(
        audit_directory, output_tag
    )
    (
        lattice_final_latex,
        final_parameter_contract,
        final_discretization_contract,
        reviewer_record,
    ) = reviewer.review(
        real_agent1_report,
        theory_contract,
        draft_parameter_contract,
        draft_discretization_contract,
        run_spec,
        reviewer_raw_file,
        reviewer_status_file,
        discretization_experience=discretization_experience,
    )
    max_local_revisions = 2
    revision_count = automatic_revision_count(
        audit_directory, "Agent2", output_tag
    )
    while reviewer_record.get("machine_chain_blocked"):
        if revision_count >= max_local_revisions:
            raise SystemExit(
                "Agent2 reviewer still reports a fatal scientific issue after "
                "{} local revisions. Inspect {}.".format(
                    max_local_revisions, reviewer_status_file
                )
            )
        initial_reviewer_status_file = reviewer_status_file
        revision_raw_file = next_generator_raw_path(audit_directory, output_tag)
        revision_number = revision_count + 1
        reviewer_record["automatic_revision"] = {
            "attempted": True,
            "policy_version": AUTOMATIC_REVISION_POLICY_VERSION,
            "attempt_number": revision_number,
            "status": "started",
            "revision_raw_response_path": os.path.abspath(revision_raw_file),
        }
        atomic_json_write(initial_reviewer_status_file, reviewer_record)
        agent2.logger.warning(
            "Agent2 reviewer 发现 fatal 问题；执行第 %d/%d 次局部关联项修正。",
            revision_number, max_local_revisions,
        )
        try:
            (
                lattice_draft_latex,
                draft_parameter_contract,
                draft_discretization_contract,
            ) = agent2.revise_from_review(
                real_agent1_report,
                theory_contract,
                run_spec,
                draft_parameter_contract,
                draft_discretization_contract,
                reviewer_record.get("findings", []),
                raw_response_path=revision_raw_file,
                previous_report=lattice_draft_latex,
            )
        except Exception as error:
            raise SystemExit(
                "Agent2 automatic revision failed deterministic validation: {}. "
                "Inspect {} and {}.".format(
                    error, initial_reviewer_status_file, revision_raw_file
                )
            )
        # The corrected, validated result becomes the paid-stage checkpoint.
        with open(draft_file, "w", encoding="utf-8") as handle:
            handle.write(lattice_draft_latex)
        save_parameter_contract(draft_parameter_contract, draft_parameter_file)
        save_machine_contract(
            draft_discretization_contract,
            draft_discretization_file,
            validate_discretization_contract,
            expected_theory_contract=theory_contract,
            expected_parameter_contract=draft_parameter_contract,
        )
        reviewer_raw_file, reviewer_status_file = next_reviewer_attempt_paths(
            audit_directory, output_tag
        )
        (
            lattice_final_latex,
            final_parameter_contract,
            final_discretization_contract,
            reviewer_record,
        ) = reviewer.review(
            real_agent1_report,
            theory_contract,
            draft_parameter_contract,
            draft_discretization_contract,
            run_spec,
            reviewer_raw_file,
            reviewer_status_file,
            discretization_experience=discretization_experience,
        )
        reviewer_record["automatic_revision"] = {
            "attempted": True,
            "policy_version": AUTOMATIC_REVISION_POLICY_VERSION,
            "attempt_number": revision_number,
            "status": "completed",
            "initial_review_status_path": os.path.abspath(
                initial_reviewer_status_file
            ),
            "revision_raw_response_path": os.path.abspath(revision_raw_file),
        }
        atomic_json_write(reviewer_status_file, reviewer_record)
        revision_count = revision_number
    contract_file = os.path.join(contract_directory, f"Agent2_ParameterContract_{output_tag}.json")
    save_parameter_contract(final_parameter_contract, contract_file)
    discretization_file = os.path.join(
        contract_directory, f"Agent2_DiscretizationContract_{output_tag}.json"
    )
    save_machine_contract(
        final_discretization_contract,
        discretization_file,
        validate_discretization_contract,
        expected_theory_contract=theory_contract,
        expected_parameter_contract=final_parameter_contract,
    )
    audit_record = publish_audit_report(
        lattice_final_latex,
        stage_name="Agent2_Verified",
        output_tag=output_tag,
        report_directory=report_directory,
        audit_directory=audit_directory,
        compiled_filename=f"Agent2_Verified_{output_tag}.tex",
        rejected_filename=f"Agent2_Rejected_{output_tag}.tex",
        status_filename=f"Agent2_AuditStatus_{output_tag}.json",
    )
    final_file = audit_record["report_path"]
    reviewer.logger.info(f"格点化终稿已保存至: {final_file}")
    reviewer.logger.info(f"参数契约已保存至: {contract_file}")
    write_stage_output_manifest(
        args.output_manifest,
        "agent2",
        {
            "draft_report": draft_file,
            "draft_parameter_contract": draft_parameter_file,
            "draft_discretization_contract": draft_discretization_file,
            "report": final_file,
            "audit_status": audit_record["status_path"],
            "parameter_contract": contract_file,
            "discretization_contract": discretization_file,
            "review_status": reviewer_status_file,
            "review_raw_response": reviewer_raw_file,
        },
    )
