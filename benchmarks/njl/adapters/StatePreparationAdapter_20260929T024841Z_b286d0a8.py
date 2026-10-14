from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


PARAMETER_CONTRACT = {'constraints': [{'applies_when': 'always',
                  'id': 'immutable_lattice_geometry',
                  'parameters': ['spatial_site_count', 'lattice_spacing'],
                  'reason': 'Preserves the supplied lattice size and every requested '
                            'coordinate while fixing the selected low-cost boundary '
                            'prescription.',
                  'requirement': 'spatial_site_count must equal 5, lattice_spacing must equal '
                                 '1, the authoritative site indices must be {-2,-1,0,1,2}, and '
                                 'the boundary condition must remain open',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'fermion_mode_relation',
                  'parameters': ['fermionic_mode_count', 'spatial_site_count'],
                  'reason': 'The selected Wilson regulator retains both Dirac components at '
                            'every site.',
                  'requirement': 'fermionic_mode_count must equal 2*spatial_site_count',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'fixed_charge_definition',
                  'parameters': ['spatial_site_count'],
                  'reason': 'This fixed regulator reference supports the independently '
                            'requested charge-zero level indices.',
                  'requirement': 'The lattice charge must be Q_lat=sum_{x,s} '
                                 'n_{x,s}-spatial_site_count, with no '
                                 'interacting-state-dependent subtraction.',
                  'severity': 'fatal'},
                 {'applies_when': 'the selected open-boundary Wilson discretization is used',
                  'id': 'wilson_endpoint_rule',
                  'parameters': ['wilson_parameter', 'lattice_spacing'],
                  'reason': 'Implements the Wilson regulator as the direct spatial second '
                            'difference.',
                  'requirement': 'The onsite Wilson coefficient must be '
                                 'wilson_parameter/lattice_spacing at every retained site, '
                                 'including both endpoints; it must not be reduced according '
                                 'to endpoint coordination number.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'unshifted_hamiltonian',
                  'parameters': [],
                  'reason': 'The continuum contract requires the unshifted Hamiltonian and '
                            'places vacuum subtraction only in the requested matrix-element '
                            'combination.',
                  'requirement': 'No additive vacuum-energy shift and no target-state '
                                 'expectation value may be inserted into the Hamiltonian or '
                                 'local interaction.',
                  'severity': 'fatal'},
                 {'applies_when': 'constructing or evolving the ordered bilocal observable',
                  'id': 'observable_sector_support',
                  'parameters': [],
                  'reason': 'Preserves the ordered operator-sector path.',
                  'requirement': 'Both Q_lat=0 and Q_lat=-1 sectors must be represented '
                                 'because the rightmost annihilation operator maps charge zero '
                                 'to charge minus one before the left insertion restores '
                                 'charge zero.',
                  'severity': 'fatal'}],
 'discretizations': [{'parameter_references': ['lattice_spacing', 'spatial_site_count'],
                      'scheme': 'centered one-dimensional spatial lattice with continuous real '
                                'time',
                      'sector': 'spacetime',
                      'settings': {'independent_constraint_count': 0,
                                   'link_count': 4,
                                   'nearest_neighbor_link_indices': [[-2, -1],
                                                                     [-1, 0],
                                                                     [0, 1],
                                                                     [1, 2]],
                                   'observable_coordinate_pullback': 't=z_index*lattice_spacing '
                                                                     'and '
                                                                     'x=z_index*lattice_spacing',
                                   'plaquette_count': 0,
                                   'spatial_boundary_condition': 'open',
                                   'spatial_dimension': 1,
                                   'spatial_site_indices': [-2, -1, 0, 1, 2],
                                   'time_representation': 'continuous Hamiltonian real time',
                                   'vertex_count': 5},
                      'truncation': {'reason': 'Contains every requested coordinate while '
                                               'avoiding a Jordan-Wigner wrap-around hopping.',
                                     'scheme': 'finite spatial volume',
                                     'settings': {'endpoint_prescription': 'open Wilson '
                                                                           'second-difference '
                                                                           'with full onsite '
                                                                           'Wilson coefficient '
                                                                           'at every retained '
                                                                           'site'}}},
                     {'parameter_references': ['bare_quark_mass',
                                               'bare_four_fermion_coupling',
                                               'lattice_spacing',
                                               'wilson_parameter',
                                               'fermionic_mode_count'],
                      'scheme': 'two-component Wilson fermions',
                      'sector': 'fermion',
                      'settings': {'alpha_1': 'sigma_3',
                                   'beta': 'sigma_1',
                                   'charge_zero_sector_dimension': 252,
                                   'components_per_site': 2,
                                   'fermionic_mode_count': 10,
                                   'finite_fock_dimension': 1024,
                                   'gamma_0': 'sigma_1',
                                   'gamma_1': '-i*sigma_2',
                                   'intermediate_charge_minus_one_sector_dimension': 210,
                                   'optional_qubit_encoding': 'Jordan-Wigner in increasing '
                                                              'site index and then '
                                                              'spinor-component order',
                                   'qubit_count': 10,
                                   'wilson_second_derivative_prescription': 'onsite r/a at '
                                                                            'every site and '
                                                                            'hopping '
                                                                            '-r*beta/(2a) on '
                                                                            'every retained '
                                                                            'nearest-neighbor '
                                                                            'link'},
                      'truncation': {'reason': 'Canonical fermion modes are exactly '
                                               'finite-dimensional; no occupation '
                                               'approximation is introduced.',
                                     'scheme': 'finite fermionic Fock space',
                                     'settings': {'occupation_per_mode': [0, 1]}}}],
 'external_states': {'definitions': {'hadron': 'Quantum-number-sector eigenstate; '
                                               'level_index=1; quantum_numbers={"charge": 0}',
                                     'vacuum': 'Quantum-number-sector eigenstate; '
                                               'level_index=0; quantum_numbers={"charge": 0}'},
                     'equivalent_groups': [],
                     'labels': ['hadron', 'vacuum']},
 'model': {'gauge_structure': 'none',
           'name': 'one flavor NJL model',
           'spacetime_dimension': '1+1'},
 'parameters': [{'domain': {'rule': 'finite real number'},
                 'name': 'bare_quark_mass',
                 'required_for_run': True,
                 'role': 'Bare mass in the continuum and Wilson lattice Hamiltonian.',
                 'selected_value': 0.5,
                 'symbol': 'm',
                 'type': 'real',
                 'unit': 'inverse_length'},
                {'domain': {'rule': 'finite real number'},
                 'name': 'bare_four_fermion_coupling',
                 'required_for_run': True,
                 'role': 'Coefficient of the direct local scalar four-fermion interaction.',
                 'selected_value': 0.5,
                 'symbol': 'g',
                 'type': 'real',
                 'unit': 'dimensionless'},
                {'domain': {'exclusive_minimum': 0},
                 'name': 'lattice_spacing',
                 'required_for_run': True,
                 'role': 'Common spatial spacing and observable coordinate-conversion scale.',
                 'selected_value': 1.0,
                 'symbol': 'a',
                 'type': 'real',
                 'unit': 'length'},
                {'domain': {'minimum': 1},
                 'name': 'spatial_site_count',
                 'required_for_run': True,
                 'role': 'Immutable number of spatial lattice sites.',
                 'selected_value': 5,
                 'symbol': 'L',
                 'type': 'integer',
                 'unit': None},
                {'domain': {'minimum': 0},
                 'name': 'wilson_parameter',
                 'required_for_run': True,
                 'role': 'Wilson doubler-removal coefficient.',
                 'selected_value': 1.0,
                 'symbol': 'r',
                 'type': 'real',
                 'unit': None},
                {'domain': {'minimum': 2},
                 'name': 'fermionic_mode_count',
                 'required_for_run': True,
                 'role': 'Derived number of canonical fermionic modes, equal to two times the '
                         'spatial site count.',
                 'selected_value': 10,
                 'symbol': 'N_f^{modes}',
                 'type': 'integer',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare coupling: g',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 0.5,
                 'source_ref': '/user_input/parameters/bare coupling: g',
                 'symbol': 'bare coupling: g',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'bare quark mass: m',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 0.5,
                 'source_ref': '/user_input/parameters/bare quark mass: m',
                 'symbol': 'bare quark mass: m',
                 'type': 'real',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'lattice spacing: a',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 1,
                 'source_ref': '/user_input/parameters/lattice spacing: a',
                 'symbol': 'lattice spacing: a',
                 'type': 'integer',
                 'unit': None},
                {'domain': {'source': 'immutable RunSpec value'},
                 'immutable': True,
                 'name': 'number of lattice site: L',
                 'required_for_run': True,
                 'role': 'Immutable user input; downstream stages must reference this path.',
                 'selected_value': 5,
                 'source_ref': '/user_input/parameters/number of lattice site: L',
                 'symbol': 'number of lattice site: L',
                 'type': 'integer',
                 'unit': None}],
 'schema_version': '1.1',
 'validation_obligations': [{'description': 'Confirm after construction that [H_lat,Q_lat]=0 '
                                            'exactly up to numerical representation tolerance.',
                             'id': 'charge_commutator_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that vacuum is level_index 0 and hadron '
                                            'is level_index 1 of the Q_lat=0 Hamiltonian '
                                            'block, without identifying the two states.',
                             'id': 'external_state_sector_and_level_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that every ordered observable insertion '
                                            'and real-time evolution is defined on the visited '
                                            'Q_lat=-1 intermediate sector and returns to '
                                            'Q_lat=0.',
                             'id': 'intermediate_sector_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that z_index=0 is represented by the '
                                            'direct finite operator a^{-1} c_0^dagger '
                                            '(I-sigma_3) c_0 and that the interaction uses the '
                                            'direct finite S_x^2 operator.',
                             'id': 'coincident_composite_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that no link variables, Gauss-law '
                                            'constraints, gauge fixing, or gauge transporter '
                                            'are introduced for this non-gauge model.',
                             'id': 'no_gauge_sector_check',
                             'severity': 'fatal',
                             'stage': 'pre_simulation'}]}


_REFERENCE_OPERATIONS_HADRON = [
    {"operation": "gate", "gate": "h", "targets": ["q0"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "cx", "targets": ["q1"], "controls": ["q0"], "angle": None},
    {"operation": "gate", "gate": "z", "targets": ["q0"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "h", "targets": ["q2"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "cx", "targets": ["q3"], "controls": ["q2"], "angle": None},
    {"operation": "gate", "gate": "z", "targets": ["q2"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "h", "targets": ["q4"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "cx", "targets": ["q5"], "controls": ["q4"], "angle": None},
    {"operation": "gate", "gate": "h", "targets": ["q6"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "cx", "targets": ["q7"], "controls": ["q6"], "angle": None},
    {"operation": "gate", "gate": "z", "targets": ["q6"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "h", "targets": ["q8"], "controls": [], "angle": None},
    {"operation": "gate", "gate": "cx", "targets": ["q9"], "controls": ["q8"], "angle": None},
    {"operation": "gate", "gate": "z", "targets": ["q8"], "controls": [], "angle": None},
]

_REFERENCE_OPERATIONS_VACUUM = [
    *_REFERENCE_OPERATIONS_HADRON[:8],
    {"operation": "gate", "gate": "z", "targets": ["q4"], "controls": [], "angle": None},
    *_REFERENCE_OPERATIONS_HADRON[8:],
]

STATE_PREPARATION_CONTRACT = {'adapter_interface': {'builder': 'build_state_preparation',
                       'returns': ['QuantumCircuit', 'ordered_parameters'],
                       'supports_controlled': True,
                       'supports_inverse': True,
                       'unitary': True},
 'ansatz': {'depth_extension_rule': 'For an existing depth D_ans circuit, append one complete '
                                    'layer with layer index d=D_ans and 27 fresh parameters; '
                                    'do not alter, reorder, or reparameterize any earlier '
                                    'layer.',
            'depth_symbol': 'D_ans',
            'family_id': 'NJL1D_Wilson10_shared_U1_local_HVA_v1',
            'layer_operations': [{'angle': 'alpha[d,j]',
                                  'controls': [],
                                  'gate': 'rz',
                                  'index_ranges': {'d': '0 <= d < D_ans',
                                                   'iteration_order': 'ascending j',
                                                   'j': '0 <= j <= 8'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q[j]']},
                                 {'angle': 'beta[d,j]',
                                  'controls': [],
                                  'gate': 'rzz',
                                  'index_ranges': {'d': '0 <= d < D_ans',
                                                   'iteration_order': 'ascending j after all '
                                                                      'rz gates of layer d',
                                                   'j': '0 <= j <= 8'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q[j]', 'q[j+1]']},
                                 {'angle': 'gamma[d,j]',
                                  'controls': [],
                                  'gate': 'rxx',
                                  'index_ranges': {'d': '0 <= d < D_ans',
                                                   'iteration_order': 'ascending j after all '
                                                                      'rzz gates of layer d; '
                                                                      'immediately precedes '
                                                                      'the matching ryy gate',
                                                   'j': '0 <= j <= 8'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q[j]', 'q[j+1]']},
                                 {'angle': 'gamma[d,j]',
                                  'controls': [],
                                  'gate': 'ryy',
                                  'index_ranges': {'d': '0 <= d < D_ans',
                                                   'iteration_order': 'apply immediately after '
                                                                      'rxx(gamma[d,j]) on the '
                                                                      'same edge, then advance '
                                                                      'j',
                                                   'j': '0 <= j <= 8'},
                                  'operation': 'gate_or_indexed_loop',
                                  'targets': ['q[j]', 'q[j+1]']}],
            'nested_by_depth': True,
            'parameter_order_rule': 'Use ascending layer index d=0,...,D_ans-1. Within each '
                                    'layer order alpha[d,0:9] for rz on q0 through q8, then '
                                    'beta[d,0:9] for rzz on edges (q0,q1) through (q8,q9), '
                                    'then gamma[d,0:9] for the shared-angle rxx followed by '
                                    'ryy exchange blocks on those same ascending edges. Qiskit '
                                    'conventions are rz(theta)=exp(-i*theta*Z/2) and '
                                    'rpp(theta)=exp(-i*theta*P tensor P/2).',
            'physical_sector': {'preserved_exactly': True,
                                'required': True,
                                'rule': 'Both references have sum_j n_j=5 and Q_qc=0. Every rz '
                                        'and rzz gate commutes with total number. For each '
                                        'edge, rxx(gamma) followed by ryy(gamma) equals '
                                        'exp[-i*gamma*(XX+YY)/2] and commutes with '
                                        'Z_j+Z_(j+1). Therefore every complete layer preserves '
                                        'the exact 252-dimensional popcount-five sector for '
                                        'arbitrary parameters.'},
            'recommended_test_depths': [1, 2, 4, 6, 8],
            'reference_state_rule': 'Initialize |n9...n0>=|1010101010>. For each site pair '
                                    '(q[2*l],q[2*l+1]), h on the even qubit followed by '
                                    'cx(even,odd) creates (|01>+|10>)/sqrt(2); a subsequent z '
                                    'on the even qubit changes it to (|01>-|10>)/sqrt(2). '
                                    'Vacuum uses the minus state on all five pairs. Hadron '
                                    'omits z only on the central pair (q4,q5). Both references '
                                    'have popcount five, are orthogonal, and receive the '
                                    'identical shared unitary with identical parameters.',
            'reference_states': {'hadron': {'basis_state': '1010101010',
                                            'operations': [{'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q0']},
                                                           {'angle': None,
                                                            'controls': ['q0'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q1']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q0']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q2']},
                                                           {'angle': None,
                                                            'controls': ['q2'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q3']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q2']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q4']},
                                                           {'angle': None,
                                                            'controls': ['q4'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q5']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q6']},
                                                           {'angle': None,
                                                            'controls': ['q6'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q7']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q6']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q8']},
                                                           {'angle': None,
                                                            'controls': ['q8'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q9']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q8']}]},
                                 'vacuum': {'basis_state': '1010101010',
                                            'operations': [{'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q0']},
                                                           {'angle': None,
                                                            'controls': ['q0'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q1']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q0']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q2']},
                                                           {'angle': None,
                                                            'controls': ['q2'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q3']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q2']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q4']},
                                                           {'angle': None,
                                                            'controls': ['q4'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q5']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q4']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q6']},
                                                           {'angle': None,
                                                            'controls': ['q6'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q7']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q6']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'h',
                                                            'operation': 'gate',
                                                            'targets': ['q8']},
                                                           {'angle': None,
                                                            'controls': ['q8'],
                                                            'gate': 'cx',
                                                            'operation': 'gate',
                                                            'targets': ['q9']},
                                                           {'angle': None,
                                                            'controls': [],
                                                            'gate': 'z',
                                                            'operation': 'gate',
                                                            'targets': ['q8']}]}},
            'symmetry_preservation': [{'conserved_quantity_id': 'global_vector_U1',
                                       'gate_level_reason': 'The references have popcount '
                                                            'five, diagonal rotations conserve '
                                                            'occupation, and each shared-angle '
                                                            'rxx-ryy block is a '
                                                            'number-preserving '
                                                            'nearest-neighbor exchange.',
                                       'preserved_exactly': True,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'exact_circuit',
                                       'target_eigenvalues': {'hadron': '0', 'vacuum': '0'}},
                                      {'conserved_quantity_id': 'fermion_parity_symmetry',
                                       'gate_level_reason': 'Exact preservation of '
                                                            'five-fermion number fixes '
                                                            'ZZZZZZZZZZ to eigenvalue '
                                                            '(-1)^5=-1 without additional '
                                                            'gates.',
                                       'preserved_exactly': True,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'exact_circuit',
                                       'target_eigenvalues': {'hadron': '-1', 'vacuum': '-1'}},
                                      {'conserved_quantity_id': 'charge_conjugation_C',
                                       'gate_level_reason': 'No charge-conjugation eigenvalue '
                                                            'defines either target, and '
                                                            'enforcing commutation with '
                                                            'YYYYYYYYYY would unnecessarily '
                                                            'constrain independent local '
                                                            'phases.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'spatial_reflection_P',
                                       'gate_level_reason': 'Reflection eigenvalues are '
                                                            'unspecified; independent '
                                                            'open-chain edge parameters are '
                                                            'retained to avoid an unjustified '
                                                            'parity restriction.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'time_reversal_T',
                                       'gate_level_reason': 'The antiunitary symmetry does not '
                                                            'define either requested sector, '
                                                            'and generic variational angles '
                                                            'need not satisfy a time-reversal '
                                                            'parameter constraint.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'combined_PT',
                                       'gate_level_reason': 'No PT eigenvalue is specified, so '
                                                            'no antiunitary hard constraint is '
                                                            'imposed.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'spatial_translations',
                                       'gate_level_reason': 'One-site translation is not an '
                                                            'exact symmetry of the '
                                                            'authoritative open-chain '
                                                            'Hamiltonian; no momentum '
                                                            'projection or translation '
                                                            'parameter sharing is used.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}},
                                      {'conserved_quantity_id': 'proper_Lorentz_transformations',
                                       'gate_level_reason': 'The finite fixed-spacing lattice '
                                                            'has no exact Lorentz-boost '
                                                            'generator, so Lorentz symmetry '
                                                            'cannot be a circuit-sector '
                                                            'constraint.',
                                       'preserved_exactly': False,
                                       'state_labels': ['hadron', 'vacuum'],
                                       'strategy': 'diagnostic_only',
                                       'target_eigenvalues': {'hadron': 'unspecified',
                                                              'vacuum': 'unspecified'}}]},
 'contract_type': 'state_preparation',
 'literature_sources': [{'applicability': 'modified: its local two-mode singlet reference '
                                          'motif preserves the required half-filled U(1) '
                                          'sector, but its periodic Schwinger-model gauge '
                                          'dynamics, momentum projection, and long-range '
                                          'ansatz are not applicable to the open Wilson-NJL '
                                          'mapping',
                         'locator': 'Section "Preparation of hadronic state", Eq. '
                                    '(reference-state equation labeled refst)',
                         'source_path': 'arXiv-2606.22602v1/GPDs_v2.tex'},
                        {'applicability': 'modified: supports a minimal localized '
                                          'charge-neutral excitation as a hadron reference, '
                                          'but the staggered Schwinger-model gauge Hamiltonian '
                                          'and adiabatic wavepacket circuit are not used',
                         'locator': 'Section "Hadron Wavepacket Preparation", initial '
                                    'localized pair state and Eq. (adiabaticWP)',
                         'source_path': 'arXiv-2401.08044v2/main_v6.tex'},
                        {'applicability': 'modified: provides evidence for scalable '
                                          'symmetry-preserving local adaptive circuits, but '
                                          'its staggered gauge discretization and operator '
                                          'sequence are incompatible with the supplied '
                                          'Wilson-NJL Hamiltonian',
                         'locator': 'Section "Building Scalable State Preparation Quantum '
                                    'Circuits using SC-ADAPT-VQE with Classical Computing", '
                                    'Table Echifmp5gp3 and associated convergence discussion',
                         'source_path': 'arXiv-2308.04481v3/Schwinger-v5.tex'}],
 'schema_version': '1.1',
 'state_labels': ['hadron', 'vacuum']}


_HAMILTONIAN_TERMS = (
    ("-1.25", "IIIIIIIIII"),
    ("0.75", "IIIIIIIIXX"),
    ("0.75", "IIIIIIIIYY"),
    ("0.25", "IIIIIIIIZZ"),
    ("0.75", "IIIIIIXXII"),
    ("0.75", "IIIIIIYYII"),
    ("0.25", "IIIIIIZZII"),
    ("0.75", "IIIIXXIIII"),
    ("0.75", "IIIIYYIIII"),
    ("0.25", "IIIIZZIIII"),
    ("0.75", "IIXXIIIIII"),
    ("0.75", "IIYYIIIIII"),
    ("0.25", "IIZZIIIIII"),
    ("0.75", "XXIIIIIIII"),
    ("0.75", "YYIIIIIIII"),
    ("0.25", "ZZIIIIIIII"),
    ("0.25", "IIIIIIIYZX"),
    ("-0.25", "IIIIIIIXZY"),
    ("-0.25", "IIIIIIYZXI"),
    ("0.25", "IIIIIIXZYI"),
    ("0.25", "IIIIIYZXII"),
    ("-0.25", "IIIIIXZYII"),
    ("-0.25", "IIIIYZXIII"),
    ("0.25", "IIIIXZYIII"),
    ("0.25", "IIIYZXIIII"),
    ("-0.25", "IIIXZYIIII"),
    ("-0.25", "IIYZXIIIII"),
    ("0.25", "IIXZYIIIII"),
    ("0.25", "IYZXIIIIII"),
    ("-0.25", "IXZYIIIIII"),
    ("-0.25", "YZXIIIIIII"),
    ("0.25", "XZYIIIIIII"),
    ("-0.25", "IIIIIIXZZX"),
    ("-0.25", "IIIIIIYZZY"),
    ("-0.25", "IIIIIIIXXI"),
    ("-0.25", "IIIIIIIYYI"),
    ("-0.25", "IIIIXZZXII"),
    ("-0.25", "IIIIYZZYII"),
    ("-0.25", "IIIIIXXIII"),
    ("-0.25", "IIIIIYYIII"),
    ("-0.25", "IIXZZXIIII"),
    ("-0.25", "IIYZZYIIII"),
    ("-0.25", "IIIXXIIIII"),
    ("-0.25", "IIIYYIIIII"),
    ("-0.25", "XZZXIIIIII"),
    ("-0.25", "YZZYIIIIII"),
    ("-0.25", "IXXIIIIIII"),
    ("-0.25", "IYYIIIIIII"),
)


def configure_runtime(parameter_contract):
    """Bind numerical Hamiltonian coefficients without redesigning the circuit."""
    global PARAMETER_CONTRACT
    global _HAMILTONIAN_TERMS

    entries = parameter_contract.get("parameters", [])
    values = {
        entry.get("name"): entry.get("selected_value")
        for entry in entries
        if isinstance(entry, dict)
    }
    required = {
        "bare_quark_mass",
        "bare_four_fermion_coupling",
        "lattice_spacing",
        "spatial_site_count",
        "wilson_parameter",
        "fermionic_mode_count",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing runtime parameters: {missing}")

    mass = float(values["bare_quark_mass"])
    coupling = float(values["bare_four_fermion_coupling"])
    spacing = float(values["lattice_spacing"])
    site_count = int(values["spatial_site_count"])
    wilson = float(values["wilson_parameter"])
    mode_count = int(values["fermionic_mode_count"])
    if spacing <= 0.0:
        raise ValueError("lattice_spacing must be positive")
    if site_count != 5 or mode_count != 10:
        raise ValueError(
            "This approved circuit program supports L=5 and ten fermion modes; "
            "a different L requires a separately approved scalable adapter."
        )

    labels = tuple(pauli for _, pauli in _HAMILTONIAN_TERMS)
    onsite_xy = (mass + wilson / spacing) / 2.0
    onsite_zz = coupling / 2.0
    derivative = 1.0 / (4.0 * spacing)
    link_wilson = -wilson / (4.0 * spacing)
    coefficients = (
        [-site_count * coupling / 2.0]
        + [value for _ in range(site_count) for value in (onsite_xy, onsite_xy, onsite_zz)]
        + [sign * derivative for _ in range(site_count - 1) for sign in (1.0, -1.0, -1.0, 1.0)]
        + [link_wilson] * (4 * (site_count - 1))
    )
    if len(labels) != len(coefficients):
        raise RuntimeError("Hamiltonian template and runtime coefficients disagree")
    _HAMILTONIAN_TERMS = tuple(
        (repr(float(coefficient)), pauli)
        for coefficient, pauli in zip(coefficients, labels)
    )
    PARAMETER_CONTRACT = parameter_contract


configure_runtime(PARAMETER_CONTRACT)


def build_hamiltonian():
    terms = [
        (pauli, complex(float(Fraction(coefficient))))
        for coefficient, pauli in _HAMILTONIAN_TERMS
    ]
    return SparsePauliOp.from_list(terms)


def physical_basis_indices(state_label=None):
    if state_label not in (None, "hadron", "vacuum"):
        raise ValueError(f"unknown state label: {state_label!r}")
    return np.fromiter(
        (basis_index for basis_index in range(1 << 10) if basis_index.bit_count() == 5),
        dtype=np.int64,
        count=252,
    )


def build_state_preparation(state_label, depth, reference_level=None):
    if state_label not in ("hadron", "vacuum"):
        raise ValueError(f"unknown state label: {state_label!r}")
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)):
        raise TypeError("depth must be a positive integer")
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be at least one")

    requested_level = 1 if state_label == "hadron" else 0
    if reference_level is None:
        selected_level = requested_level
    else:
        if isinstance(reference_level, (bool, np.bool_)) or not isinstance(
            reference_level, (int, np.integer)
        ):
            raise TypeError("reference_level must be an integer or None")
        selected_level = int(reference_level)
        if selected_level < 0 or selected_level > requested_level:
            raise ValueError(
                f"reference_level must be between 0 and {requested_level}"
            )

    alpha = ParameterVector("alpha", 9 * depth)
    beta = ParameterVector("beta", 9 * depth)
    gamma = ParameterVector("gamma", 9 * depth)

    circuit = QuantumCircuit(10, name=f"{state_label}_state_preparation")

    for odd_qubit in (1, 3, 5, 7, 9):
        circuit.x(odd_qubit)

    for even_qubit in (0, 2, 4, 6, 8):
        circuit.h(even_qubit)
        circuit.cx(even_qubit, even_qubit + 1)
        if selected_level == 0 or even_qubit != 4:
            circuit.z(even_qubit)

    ordered_parameters = []
    for layer in range(depth):
        offset = 9 * layer

        layer_alpha = [alpha[offset + edge] for edge in range(9)]
        layer_beta = [beta[offset + edge] for edge in range(9)]
        layer_gamma = [gamma[offset + edge] for edge in range(9)]
        ordered_parameters.extend(layer_alpha)
        ordered_parameters.extend(layer_beta)
        ordered_parameters.extend(layer_gamma)

        for qubit, parameter in enumerate(layer_alpha):
            circuit.rz(parameter, qubit)

        for edge, parameter in enumerate(layer_beta):
            circuit.rzz(parameter, edge, edge + 1)

        for edge, parameter in enumerate(layer_gamma):
            circuit.rxx(parameter, edge, edge + 1)
            circuit.ryy(parameter, edge, edge + 1)

    circuit.global_phase = 0.0
    return circuit, ordered_parameters
