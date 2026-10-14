from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


PARAMETER_CONTRACT = {'constraints': [{'applies_when': 'always',
                  'id': 'positive_hamiltonian_scales',
                  'parameters': ['bare_coupling_g', 'lattice_spacing_a'],
                  'reason': 'The Hamiltonian contains g^(-2) and a^(-1).',
                  'requirement': 'bare_coupling_g > 0 and lattice_spacing_a > 0',
                  'severity': 'fatal'},
                 {'applies_when': 'the selected exact plaquette-dual gauge scheme is used',
                  'id': 'open_boundary_dual_reduction',
                  'parameters': [],
                  'reason': 'The selected one-qubit-per-plaquette reduction contains no '
                            'independent noncontractible holonomy degrees of freedom.',
                  'requirement': 'The spatial boundary condition must be open and the lattice '
                                 'must be connected and simply connected.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing either external state',
                  'id': 'physical_gauss_sector',
                  'parameters': [],
                  'reason': "Gauss's law zero denotes zero gauge charge, not eigenvalue zero.",
                  'requirement': 'The state must satisfy G_v=+1 at every lattice vertex; in '
                                 'the reduced representation each eliminated G_v must act as '
                                 'the identity.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing the glueball state',
                  'id': 'glueball_level_preservation',
                  'parameters': [],
                  'reason': 'The external-state level index is authoritative.',
                  'requirement': 'Use zero-based level_index=1 within the G_v=+1 physical '
                                 'sector and do not replace it by the sector ground state.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing the Hamiltonian',
                  'id': 'no_vacuum_energy_shift',
                  'parameters': ['bare_coupling_g', 'lattice_spacing_a'],
                  'reason': 'Vacuum subtraction belongs only to the requested matrix-element '
                            'combination.',
                  'requirement': 'Use the unshifted Hamiltonian with coefficients determined '
                                 'only by bare_coupling_g and lattice_spacing_a; do not add or '
                                 'subtract a target-state expectation value.',
                  'severity': 'fatal'}],
 'discretizations': [{'parameter_references': ['lattice_spacing_a'],
                      'scheme': 'finite square spatial lattice with continuous Hamiltonian '
                                'time',
                      'sector': 'spacetime',
                      'settings': {'horizontal_link_indices': {'x': [0, 2], 'y': [0, 3]},
                                   'independent_gauss_constraint_count': 15,
                                   'link_count': 24,
                                   'plaquette_count': 9,
                                   'plaquette_indices': {'x': [0, 2], 'y': [0, 2]},
                                   'spatial_boundary_condition': 'open',
                                   'spatial_shape': [4, 4],
                                   'time_realization': 'continuous',
                                   'vertex_count': 16,
                                   'vertex_indices': {'x': [0, 3], 'y': [0, 3]},
                                   'vertical_link_indices': {'x': [0, 3], 'y': [0, 2]}},
                      'truncation': {'reason': 'The theory is intrinsically a finite '
                                               'spatial-lattice Hamiltonian theory with '
                                               'continuous time.',
                                     'scheme': 'none',
                                     'settings': {}}},
                     {'parameter_references': ['bare_coupling_g', 'lattice_spacing_a'],
                      'scheme': 'exact Gauss-law reduction to open-boundary plaquette-dual '
                                'variables',
                      'sector': 'gauge',
                      'settings': {'dual_degrees_of_freedom': 'one Pauli qubit tau_p for each '
                                                              'plaquette defined by the '
                                                              'spacetime settings',
                                   'effective_qubit_count': 9,
                                   'electric_mapping': 'X_l maps to the product of tau_p^x '
                                                       'over plaquettes incident on link l',
                                   'explicit_embedding_complex128_bytes': 137438953472,
                                   'full_dense_complex128_bytes': 4503599627370496,
                                   'gauge_fixing': 'none',
                                   'gauge_group': 'Z2',
                                   'gauss_sector': 'G_v=+1 at every lattice vertex',
                                   'maximum_reduced_term_support': 2,
                                   'original_kinematic_dimension': 16777216,
                                   'original_link_local_dimension': 2,
                                   'physical_dimension': 512,
                                   'plaquette_mapping': 'B_p maps to tau_p^z',
                                   'reduced_dense_complex128_bytes': 4194304},
                      'truncation': {'reason': 'Z2 links are intrinsically finite-dimensional, '
                                               'and the plaquette-dual reduction is exactly '
                                               'unitarily equivalent to the full charge-free '
                                               'physical sector under OBC.',
                                     'scheme': 'none',
                                     'settings': {'finite_group_representation': 'complete '
                                                                                 'two-dimensional '
                                                                                 'Z2 link '
                                                                                 'representation',
                                                  'reduction_exact': True}}}],
 'external_states': {'definitions': {'glueball': 'Quantum-number-sector eigenstate; '
                                                 'level_index=1; quantum_numbers={"Gauss\'s '
                                                 'law": 0}',
                                     'vacuum': 'Ground state / vacuum'},
                     'equivalent_groups': [],
                     'labels': ['glueball', 'vacuum']},
 'model': {'gauge_structure': 'local Z2',
           'name': '2+1 dimensional pure Z2 gauge theory in Kogut-Susskind formalism',
           'spacetime_dimension': '2+1'},
 'parameters': [{'domain': {'minimum_exclusive': 0},
                 'name': 'bare_coupling_g',
                 'required_for_run': True,
                 'role': 'Bare coupling in the coefficients g^2 and g^(-2).',
                 'selected_value': 1,
                 'symbol': 'g',
                 'type': 'real',
                 'unit': None},
                {'domain': {'minimum_exclusive': 0},
                 'name': 'lattice_spacing_a',
                 'required_for_run': True,
                 'role': 'Spatial lattice spacing and inverse overall Hamiltonian energy '
                         'scale; temporal spacing is the same if a downstream time '
                         'discretization is introduced.',
                 'selected_value': 1,
                 'symbol': 'a',
                 'type': 'real',
                 'unit': 'chosen natural length unit'},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare copuling: g',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 'g = 1',
                 'source_ref': '/user_input/parameters/bare copuling: g',
                 'symbol': 'bare copuling: g',
                 'type': 'string',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'number of lattice site [L,L]:',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': [4, 4],
                 'source_ref': '/user_input/parameters/number of lattice site [L,L]:',
                 'symbol': 'number of lattice site [L,L]:',
                 'type': 'list',
                 'unit': None}],
 'schema_version': '1.1',
 'validation_obligations': [{'description': 'Confirm that both named external states lie in '
                                            'the zero-charge sector G_v=+1 at every lattice '
                                            'vertex, or equivalently in the exact reduced '
                                            'physical Hilbert space.',
                             'id': 'validate_physical_sector',
                             'severity': 'fatal',
                             'stage': 'pre_simulation'},
                            {'description': 'Confirm that the reduced plaquette and electric '
                                            'operators reproduce the physical-sector algebra '
                                            'of B_p and X_l, including all '
                                            'link-to-incident-plaquette relations at the open '
                                            'boundary.',
                             'id': 'validate_dual_operator_algebra',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that glueball is the zero-based level-1 '
                                            'Hamiltonian eigenstate in the physical Gauss '
                                            'sector, without imposing an unspecified D4 or '
                                            'other discrete-symmetry eigenvalue.',
                             'id': 'validate_glueball_level',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that no vacuum-energy subtraction or '
                                            'state-dependent coefficient has been inserted '
                                            'into the constructed Hamiltonian.',
                             'id': 'validate_unshifted_hamiltonian',
                             'severity': 'fatal',
                             'stage': 'post_construction'}]}


STATE_PREPARATION_CONTRACT = {'adapter_interface': {'builder': 'build_state_preparation',
                       'returns': ['QuantumCircuit', 'ordered_parameters'],
                       'supports_controlled': False,
                       'supports_inverse': True,
                       'unitary': True},
 'ansatz': {'depth_extension_rule': 'For each state label independently, prepare its fixed '
                                    'reference and then apply complete layers ell=1,...,D_ans '
                                    'in increasing circuit-time order. Extend depth D_ans to '
                                    'D_ans+1 by appending one complete layer with 29 fresh '
                                    'parameters. The depth-D_ans circuit is therefore an exact '
                                    'prefix of the depth-(D_ans+1) circuit. For the present '
                                    'nine-qubit grid, one layer contains 9 RZ gates, 8 RX '
                                    'gates, and 12 RXX gates. Using RXX(phi) = (H tensor H) CX '
                                    'RZ(phi) CX (H tensor H), a layer contains 24 CX gates and '
                                    '77 one-qubit gates before cancellation. Every '
                                    'parameterized primitive has control arity zero.',
            'depth_symbol': 'D_ans',
            'family_id': 'independent_term_open_z2_dual_hamiltonian_variational_ansatz',
            'layer_operations': [{'angle': '-2*alpha_Z[ell,q_index]',
                                  'controls': [],
                                  'gate': 'RZGate',
                                  'index_ranges': {'ell': '1..D_ans in increasing order',
                                                   'q_index': '0,1,2,3,4,5,6,7,8 in increasing '
                                                              'order'},
                                  'operation': 'indexed_loop',
                                  'targets': ['q[q_index]']},
                                 {'angle': '-2*boundary_weight[q_index]*alpha_X[ell,q_index]',
                                  'controls': [],
                                  'gate': 'RXGate',
                                  'index_ranges': {'boundary_weight': 'boundary_weight[0]=2, '
                                                                      'boundary_weight[1]=1, '
                                                                      'boundary_weight[2]=2, '
                                                                      'boundary_weight[3]=1, '
                                                                      'boundary_weight[5]=1, '
                                                                      'boundary_weight[6]=2, '
                                                                      'boundary_weight[7]=1, '
                                                                      'boundary_weight[8]=2',
                                                   'ell': '1..D_ans in increasing order',
                                                   'q_index': '0,1,2,3,5,6,7,8 in increasing '
                                                              'order'},
                                  'operation': 'indexed_loop',
                                  'targets': ['q[q_index]']},
                                 {'angle': '-2*alpha_XX[ell,edge_first,edge_second]',
                                  'controls': [],
                                  'gate': 'RXXGate',
                                  'index_ranges': {'ell': '1..D_ans in increasing order',
                                                   'ordered_edges': '(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,7),(5,8),(6,7),(7,8)'},
                                  'operation': 'indexed_loop',
                                  'targets': ['q[edge_first]', 'q[edge_second]']}],
            'nested_by_depth': True,
            'parameter_order_rule': 'Glueball and vacuum use separate parameter objects and '
                                    'parameter vectors; no parameter is shared between labels. '
                                    'Within either label, order first by layer ell=1,...,D_ans '
                                    'and within each layer as [alpha_Z[ell,0], alpha_Z[ell,1], '
                                    'alpha_Z[ell,2], alpha_Z[ell,3], alpha_Z[ell,4], '
                                    'alpha_Z[ell,5], alpha_Z[ell,6], alpha_Z[ell,7], '
                                    'alpha_Z[ell,8], alpha_X[ell,0], alpha_X[ell,1], '
                                    'alpha_X[ell,2], alpha_X[ell,3], alpha_X[ell,5], '
                                    'alpha_X[ell,6], alpha_X[ell,7], alpha_X[ell,8], '
                                    'alpha_XX[ell,0,1], alpha_XX[ell,0,3], alpha_XX[ell,1,2], '
                                    'alpha_XX[ell,1,4], alpha_XX[ell,2,5], alpha_XX[ell,3,4], '
                                    'alpha_XX[ell,3,6], alpha_XX[ell,4,5], alpha_XX[ell,4,7], '
                                    'alpha_XX[ell,5,8], alpha_XX[ell,6,7], alpha_XX[ell,7,8]]. '
                                    'Qiskit conventions are RZ(phi)=exp(-i*phi*Z/2), '
                                    'RX(phi)=exp(-i*phi*X/2), and RXX(phi)=exp(-i*phi*X tensor '
                                    'X/2). Implemented angles are -2*alpha_Z on every qubit; '
                                    '-4*alpha_X on corner qubits 0,2,6,8; -2*alpha_X on '
                                    'edge-noncorner qubits 1,3,5,7; and -2*alpha_XX on every '
                                    'listed nearest-neighbour edge. These factors retain the '
                                    'exact aggregated Hamiltonian coefficients -1 for Z, -2 '
                                    'for corner X, -1 for noncorner boundary X, and -1 for XX.',
            'physical_sector': {'preserved_exactly': True,
                                'required': True,
                                'rule': 'The complete 512-dimensional nine-qubit Hilbert space '
                                        'is the exact charge-free physical space after the '
                                        'open-boundary plaquette-dual reduction. For every one '
                                        'of the 16 supplied vertex generators, the ordered '
                                        'product of mapped link operators is IIIIIIIII with '
                                        'eigenvalue +1. Therefore both references and every '
                                        'RZ, RX, and RXX layer remain exactly in the G_v=+1 '
                                        'sector without penalties or basis filtering.'},
            'recommended_test_depths': [1, 2, 4, 8],
            'reference_state_rule': 'Computational bitstrings are written as |q8 q7 q6 q5 q4 '
                                    'q3 q2 q1 q0>, where q[x+3*y] is plaquette p(x,y). Both '
                                    'circuits initialize |000000000>. Vacuum applies H to q0 '
                                    'through q8 and prepares |+> tensor 9. Glueball applies X '
                                    'to central plaquette q4 before applying H to q0 through '
                                    'q8, preparing q4 in |-> and every other qubit in |+>. The '
                                    'two reference states are orthogonal. Each label then '
                                    'receives its own independent variational parameters; the '
                                    'glueball is targeted as zero-based distinct-energy level '
                                    '1 through the external sequential lower-level '
                                    'orthogonality procedure, without supplying its exact '
                                    'energy or imposing a D4 sector.',
            'reference_states': {'glueball': {'basis_state': '000000000 in |q8 q7 q6 q5 q4 q3 '
                                                             'q2 q1 q0> order',
                                              'operations': [{'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'XGate',
                                                              'operation': 'gate',
                                                              'targets': ['q4']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q0']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q1']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q2']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q3']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q4']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q5']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q6']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q7']},
                                                             {'angle': 'not_applicable_fixed_gate',
                                                              'controls': [],
                                                              'gate': 'HGate',
                                                              'operation': 'gate',
                                                              'targets': ['q8']}]},
                                 'vacuum': {'basis_state': '000000000 in |q8 q7 q6 q5 q4 q3 q2 '
                                                           'q1 q0> order',
                                            'operations': [{'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q0']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q1']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q2']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q3']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q4']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q5']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q6']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q7']},
                                                           {'angle': 'not_applicable_fixed_gate',
                                                            'controls': [],
                                                            'gate': 'HGate',
                                                            'operation': 'gate',
                                                            'targets': ['q8']}]}},
            'symmetry_preservation': [{'conserved_quantity_id': 'local_z2_gauge_symmetry',
                                       'gate_level_reason': 'Every mapped Gauss generator is '
                                                            'IIIIIIIII on the reduced '
                                                            'register, so every reference gate '
                                                            'and every parameterized gate '
                                                            'commutes with it. Exact '
                                                            'preservation has zero gate '
                                                            'overhead and leaves the full '
                                                            '512-dimensional physical space '
                                                            'available, so it does not '
                                                            'restrict access to either '
                                                            'requested energy level.',
                                       'preserved_exactly': True,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'exact_circuit',
                                       'target_eigenvalues': {'glueball': '+1 at every lattice '
                                                                          'vertex',
                                                              'vacuum': '+1 at every lattice '
                                                                        'vertex'}},
                                      {'conserved_quantity_id': 'continuous_time_translation',
                                       'gate_level_reason': 'Energy labels specify the '
                                                            'optimized output eigenstates, not '
                                                            'a gatewise symmetry sector. '
                                                            'Requiring all gates to commute '
                                                            'with H_qc would preserve the '
                                                            "references' energy distributions "
                                                            'and prevent preparation of the '
                                                            'desired eigenstates.',
                                       'preserved_exactly': False,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'glueball': 'zero-based level-1 '
                                                                          'distinct energy in '
                                                                          'the physical sector',
                                                              'vacuum': 'ground-state energy'}},
                                      {'conserved_quantity_id': 'time_reversal',
                                       'gate_level_reason': 'No time-reversal eigenvalue is '
                                                            'provided. General RZ, RX, and RXX '
                                                            'angles can produce complex '
                                                            'amplitudes, so K symmetry is not '
                                                            'imposed as a circuit restriction.',
                                       'preserved_exactly': False,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'glueball': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'open_square_dihedral_symmetry',
                                       'gate_level_reason': 'No D4 irrep or eigenvalue is '
                                                            'specified. Independent site and '
                                                            'edge parameters intentionally '
                                                            'avoid enforcing D4, ensuring that '
                                                            'no component of the authoritative '
                                                            'level-1 eigenspace is excluded.',
                                       'preserved_exactly': False,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'glueball': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'spatial_translation_symmetry',
                                       'gate_level_reason': 'The open boundary breaks every '
                                                            'nonidentity spatial translation, '
                                                            'so no translation constraint is '
                                                            'imposed on either reference or '
                                                            'layer.',
                                       'preserved_exactly': False,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'glueball': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'nontrivial_electric_z2_one_form_symmetry',
                                       'gate_level_reason': 'Open boundaries provide no '
                                                            'noncontractible one-form '
                                                            'generator. Contractible curve '
                                                            'products reduce to Gauss '
                                                            'identities already preserved by '
                                                            'the exact physical encoding, so '
                                                            'no additional circuit constraint '
                                                            'is imposed.',
                                       'preserved_exactly': False,
                                       'state_labels': ['glueball', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'glueball': 'unspecified',
                                                              'vacuum': 'unspecified'}}]},
 'contract_type': 'state_preparation',
 'literature_sources': [{'applicability': 'modified: the HVA is applied to the authoritative '
                                          'open-boundary dual Z2 Hamiltonian, with separate '
                                          'parameters for every unique Z, boundary-X, and '
                                          'nearest-neighbour-XX generator.',
                         'locator': 'Hamiltonian Variational Ansatz definition, Eq. (hva)',
                         'source_path': 'arXiv-2008.02941v2.tar/main.tex'},
                        {'applicability': 'incompatible as the fixed handoff ansatz: it '
                                          'requires adaptive operator selection and does not '
                                          'supply a predetermined scalable gate sequence for '
                                          'this nine-qubit Z2 dual Hamiltonian.',
                         'locator': 'Section II, SC-ADAPT-VQE algorithm steps 1--6',
                         'source_path': 'arXiv-2308.04481v3/Schwinger-v5.tex'},
                        {'applicability': 'incompatible as the selected family: the generic '
                                          'hardware-efficient alternative does not encode the '
                                          'supplied interaction graph and offers no '
                                          'physical-sector or model-specific advantage over '
                                          'the HVA.',
                         'locator': 'Discussion of chemistry-inspired, ADAPT-VQE, and '
                                    'hardware-based ansatz choices preceding the reported '
                                    'ansatz comparisons',
                         'source_path': 'arXiv-1904.10910v3.tar/main2.tex'}],
 'schema_version': '1.1',
 'state_labels': ['glueball', 'vacuum']}


_NUM_QUBITS = 9
_STATE_LABELS = ("glueball", "vacuum")
_X_QUBITS = (0, 1, 2, 3, 5, 6, 7, 8)
_X_WEIGHTS = (2, 1, 2, 1, 1, 2, 1, 2)
_XX_EDGES = (
    (0, 1),
    (0, 3),
    (1, 2),
    (1, 4),
    (2, 5),
    (3, 4),
    (3, 6),
    (4, 5),
    (4, 7),
    (5, 8),
    (6, 7),
    (7, 8),
)
_HAMILTONIAN_LABELS = (
    "IIIIIIIIX",
    "IIIIIXIIX",
    "IIXIIXIII",
    "IIXIIIIII",
    "IIIIIIIXI",
    "IIIIXIIXI",
    "IXIIXIIII",
    "IXIIIIIII",
    "IIIIIIXII",
    "IIIXIIXII",
    "XIIXIIIII",
    "XIIIIIIII",
    "IIIIIIIIX",
    "IIIIIXIII",
    "IIXIIIIII",
    "IIIIIIIXX",
    "IIIIXXIII",
    "IXXIIIIII",
    "IIIIIIXXI",
    "IIIXXIIII",
    "XXIIIIIII",
    "IIIIIIXII",
    "IIIXIIIII",
    "XIIIIIIII",
    "IIIIIIIIZ",
    "IIIIIIIZI",
    "IIIIIIZII",
    "IIIIIZIII",
    "IIIIZIIII",
    "IIIZIIIII",
    "IIZIIIIII",
    "IZIIIIIII",
    "ZIIIIIIII",
)


def build_hamiltonian():
    return SparsePauliOp.from_list(
        [(pauli_label, -1.0) for pauli_label in _HAMILTONIAN_LABELS]
    )


def physical_basis_indices(state_label=None):
    if state_label is not None and state_label not in _STATE_LABELS:
        raise ValueError(f"Unknown state label: {state_label!r}")
    return np.arange(1 << _NUM_QUBITS, dtype=np.int64)


def build_state_preparation(state_label, depth):
    if state_label not in _STATE_LABELS:
        raise ValueError(f"Unknown state label: {state_label!r}")
    if type(depth) is not int:
        raise TypeError("depth must be a positive integer")
    if depth < 1:
        raise ValueError("depth must be at least one")

    circuit = QuantumCircuit(_NUM_QUBITS, name=f"{state_label}_state_preparation")

    if state_label == "glueball":
        circuit.x(4)
    for qubit in range(_NUM_QUBITS):
        circuit.h(qubit)

    alpha_z = ParameterVector(f"{state_label}_alpha_Z", depth * 9)
    alpha_x = ParameterVector(f"{state_label}_alpha_X", depth * 8)
    alpha_xx = ParameterVector(f"{state_label}_alpha_XX", depth * 12)
    ordered_parameters = []

    for layer in range(depth):
        z_offset = layer * 9
        x_offset = layer * 8
        xx_offset = layer * 12

        for qubit in range(9):
            parameter = alpha_z[z_offset + qubit]
            ordered_parameters.append(parameter)
            circuit.rz(-2 * parameter, qubit)

        for local_index, (qubit, weight) in enumerate(zip(_X_QUBITS, _X_WEIGHTS)):
            parameter = alpha_x[x_offset + local_index]
            ordered_parameters.append(parameter)
            circuit.rx(-2 * weight * parameter, qubit)

        for local_index, (first, second) in enumerate(_XX_EDGES):
            parameter = alpha_xx[xx_offset + local_index]
            ordered_parameters.append(parameter)
            circuit.rxx(-2 * parameter, first, second)

    return circuit, ordered_parameters
