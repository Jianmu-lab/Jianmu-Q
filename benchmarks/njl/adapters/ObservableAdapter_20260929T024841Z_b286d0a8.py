import math
from typing import Any

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis.unitary.qsd import qs_decomposition


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


OBSERVABLE_CONTRACT = {'adapter_interface': {'builder': 'build_observable_measurements'},
 'contract_type': 'observable_measurement',
 'literature_sources': [{'applicability': 'incompatible for the measurement circuit itself: it '
                                          'supports finite product-formula real-time execution '
                                          'but does not provide the unequal-time same-state '
                                          'interferometer required here',
                         'locator': 'section "Real-time Simulations using IBM\'s Digital '
                                    'Quantum Computers", especially the paragraph defining '
                                    'forward-backward mitigation evolution',
                         'source_path': 'arXiv-2401.08044v2/main_v6.tex'}],
 'measurement': {'ancilla_qubits': 1,
                 'circuit_recipe_ids': ['hadron_k0_real',
                                        'hadron_k0_imag',
                                        'hadron_k1_real',
                                        'hadron_k1_imag',
                                        'hadron_k2_real',
                                        'hadron_k2_imag',
                                        'hadron_k3_real',
                                        'hadron_k3_imag',
                                        'vacuum_k0_real',
                                        'vacuum_k0_imag',
                                        'vacuum_k1_real',
                                        'vacuum_k1_imag',
                                        'vacuum_k2_real',
                                        'vacuum_k2_imag',
                                        'vacuum_k3_real',
                                        'vacuum_k3_imag',
                                        'hadron_z0_Z5',
                                        'vacuum_z0_Z5'],
                 'circuit_recipes': [{'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'hadron',
                                                                   'trajectory': 0},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k0_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'hadron',
                                                                   'trajectory': 0},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k0_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'hadron',
                                                                   'trajectory': 1},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k1_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'hadron',
                                                                   'trajectory': 1},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k1_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'hadron',
                                                                   'trajectory': 2},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k2_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'hadron',
                                                                   'trajectory': 2},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k2_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'hadron',
                                                                   'trajectory': 3},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k3_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'hadron',
                                                                   'trajectory': 3},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'hadron_k3_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 0},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k0_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 0},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k0_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 1},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k1_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 1},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k1_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 2},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k2_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 2},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k2_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'real',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 3},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k3_real'},
                                     {'component': 'imag',
                                      'operations': [{'activation': 'z_index in {-2,-1,1,2}',
                                                      'bindings': {'component': 'imag',
                                                                   'state': 'vacuum',
                                                                   'trajectory': 3},
                                                      'operation': 'instantiate',
                                                      'reference': 'same_state_interferometer'}],
                                      'recipe_id': 'vacuum_k3_imag'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index=0',
                                                      'bindings': {'state': 'hadron'},
                                                      'operation': 'instantiate',
                                                      'reference': 'coincident_Z5_direct'}],
                                      'recipe_id': 'hadron_z0_Z5'},
                                     {'component': 'real',
                                      'operations': [{'activation': 'z_index=0',
                                                      'bindings': {'state': 'vacuum'},
                                                      'operation': 'instantiate',
                                                      'reference': 'coincident_Z5_direct'}],
                                      'recipe_id': 'vacuum_z0_Z5'}],
                 'components': ['real', 'imag'],
                 'protocol_id': 'njl5_same_state_uncontrolled_evolution_interferometer_v1'},
 'operator_recipes': [{'id': 'endpoint_A_from_z_and_trajectory',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PAULI_STRING',
                                       'operation': 'lookup_exact_pauli',
                                       'targets': ['if z_index=-2: k=0,1 use IIIIIIIIXZ; k=2,3 '
                                                   'use IIIIIIIIYZ',
                                                   'if z_index=-1: k=0,1 use IIIIIIXZZZ; k=2,3 '
                                                   'use IIIIIIYZZZ',
                                                   'if z_index=1: k=0,1 use IIXZZZZZZZ; k=2,3 '
                                                   'use IIYZZZZZZZ',
                                                   'if z_index=2: k=0,1 use XZZZZZZZZZ; k=2,3 '
                                                   'use YZZZZZZZZZ']}]},
                      {'id': 'endpoint_B_from_trajectory',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PAULI_STRING',
                                       'operation': 'lookup_exact_pauli',
                                       'targets': ['k=0,2 use IIIIXZZZZZ',
                                                   'k=1,3 use IIIIYZZZZZ']}]},
                      {'id': 'controlled_pauli_factorwise',
                       'operations': [{'angle': None,
                                       'controls': ['measurement_ancilla'],
                                       'gate': 'CX',
                                       'operation': 'for_each_nonidentity_factor_X',
                                       'targets': ['factor qubit']},
                                      {'angle': None,
                                       'controls': ['measurement_ancilla'],
                                       'gate': 'CY_AS_SDG_CX_S',
                                       'operation': 'for_each_nonidentity_factor_Y',
                                       'targets': ['factor qubit']},
                                      {'angle': None,
                                       'controls': ['measurement_ancilla'],
                                       'gate': 'CZ',
                                       'operation': 'for_each_nonidentity_factor_Z',
                                       'targets': ['factor qubit']}]},
                      {'id': 'onsite_block_h_on_s',
                       'operations': [{'angle': 'h_on(s)=((bare_quark_mass+wilson_parameter/lattice_spacing)/2)*(X_(2s)X_(2s+1)+Y_(2s)Y_(2s+1))+(bare_four_fermion_coupling/2)*Z_(2s)Z_(2s+1), '
                                                's=0,1,2,3,4',
                                       'controls': [],
                                       'gate': 'PAULI_SUM',
                                       'operation': 'define_local_generator',
                                       'targets': ['q_(2s)', 'q_(2s+1)']}]},
                      {'id': 'link_block_h_link_l',
                       'operations': [{'angle': 'h_link(l)=(1/(4*lattice_spacing))*(X_b '
                                                'Z_(b+1) Y_(b+2)-Y_b Z_(b+1) X_(b+2)-X_(b+1) '
                                                'Z_(b+2) Y_(b+3)+Y_(b+1) Z_(b+2) '
                                                'X_(b+3))-(wilson_parameter/(4*lattice_spacing))*(X_b '
                                                'Z_(b+1) Z_(b+2) X_(b+3)+Y_b Z_(b+1) Z_(b+2) '
                                                'Y_(b+3)+X_(b+1)X_(b+2)+Y_(b+1)Y_(b+2))',
                                       'controls': [],
                                       'gate': 'PAULI_SUM',
                                       'operation': 'define_local_generator',
                                       'targets': ['q_b',
                                                   'q_(b+1)',
                                                   'q_(b+2)',
                                                   'q_(b+3)',
                                                   'b=2*l, l=0,1,2,3']}]},
                      {'id': 'exact_local_block_exponential',
                       'operations': [{'angle': 'V=exp(-i*tau*h_local), with h_local exactly '
                                                'onsite_block_h_on_s or link_block_h_link_l',
                                       'controls': [],
                                       'gate': 'MATRIX_EXPONENTIAL',
                                       'operation': 'construct_fixed_width_matrix',
                                       'targets': ['the two or four qubits of the selected '
                                                   'local block']},
                                      {'angle': 'all RZ and RY angles are deterministically '
                                                'obtained from the exact matrix V; no opaque '
                                                'evolution primitive remains',
                                       'controls': [],
                                       'gate': 'QUANTUM_SHANNON_DECOMPOSITION_TO_RZ_RY_CX',
                                       'operation': 'finite_exact_synthesis',
                                       'targets': ['the same ordered local qubits']}]},
                      {'id': 'same_state_interferometer',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'U_state',
                                       'operation': 'compose_bound_state_preparation',
                                       'targets': ['q0',
                                                   'q1',
                                                   'q2',
                                                   'q3',
                                                   'q4',
                                                   'q5',
                                                   'q6',
                                                   'q7',
                                                   'q8',
                                                   'q9']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'H',
                                       'operation': 'gate',
                                       'targets': ['measurement_ancilla']},
                                      {'angle': None,
                                       'controls': ['measurement_ancilla'],
                                       'gate': 'controlled_pauli_factorwise',
                                       'operation': 'compose',
                                       'targets': ['B_k from endpoint_B_from_trajectory']},
                                      {'angle': 'total time z_index*lattice_spacing, N_T steps',
                                       'controls': [],
                                       'gate': 'S2_PRODUCT_FORMULA_EVOLUTION',
                                       'operation': 'compose',
                                       'targets': ['q0',
                                                   'q1',
                                                   'q2',
                                                   'q3',
                                                   'q4',
                                                   'q5',
                                                   'q6',
                                                   'q7',
                                                   'q8',
                                                   'q9']},
                                      {'angle': None,
                                       'controls': ['measurement_ancilla'],
                                       'gate': 'controlled_pauli_factorwise',
                                       'operation': 'compose',
                                       'targets': ['A_k(z_index) from '
                                                   'endpoint_A_from_z_and_trajectory']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'H',
                                       'operation': 'if_component_real',
                                       'targets': ['measurement_ancilla']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'SDG',
                                       'operation': 'if_component_imag_first',
                                       'targets': ['measurement_ancilla']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'H',
                                       'operation': 'if_component_imag_second',
                                       'targets': ['measurement_ancilla']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'READOUT_PAULI_Z',
                                       'operation': 'estimator_readout',
                                       'targets': ['measurement_ancilla']}]},
                      {'id': 'coincident_Z5_direct',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'U_state',
                                       'operation': 'compose_bound_state_preparation',
                                       'targets': ['q0',
                                                   'q1',
                                                   'q2',
                                                   'q3',
                                                   'q4',
                                                   'q5',
                                                   'q6',
                                                   'q7',
                                                   'q8',
                                                   'q9']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'READOUT_PAULI',
                                       'operation': 'estimator_readout',
                                       'targets': ['IIIIZIIIII']}]}],
 'reconstruction': {'output_components': ['real', 'imag'],
                    'rule': 'Validate sample_values has exactly one key z_index with value in '
                            '{-2,-1,0,1,2}. If z_index!=0, define '
                            'R_state,k=expectation(state_k_real), '
                            'I_state,k=expectation(state_k_imag), s_hadron=+1, s_vacuum=-1. '
                            'Return Re=sum_state '
                            's_state*(R_state,0-I_state,1+I_state,2+R_state,3)/2 and '
                            'Im=sum_state s_state*(I_state,0+R_state,1-R_state,2+I_state,3)/2. '
                            'If z_index=0, return '
                            'Re=expectation(vacuum_z0_Z5)-expectation(hadron_z0_Z5) and Im=0. '
                            'Identity contributions cancel exactly between the hadron and '
                            'vacuum coefficients.',
                    'terms': [{'coefficient': 'for z_index!=0: Re +1/2; Im 0',
                               'recipe_id': 'hadron_k0_real'},
                              {'coefficient': 'for z_index!=0: Re 0; Im +1/2',
                               'recipe_id': 'hadron_k0_imag'},
                              {'coefficient': 'for z_index!=0: Re 0; Im +1/2',
                               'recipe_id': 'hadron_k1_real'},
                              {'coefficient': 'for z_index!=0: Re -1/2; Im 0',
                               'recipe_id': 'hadron_k1_imag'},
                              {'coefficient': 'for z_index!=0: Re 0; Im -1/2',
                               'recipe_id': 'hadron_k2_real'},
                              {'coefficient': 'for z_index!=0: Re +1/2; Im 0',
                               'recipe_id': 'hadron_k2_imag'},
                              {'coefficient': 'for z_index!=0: Re +1/2; Im 0',
                               'recipe_id': 'hadron_k3_real'},
                              {'coefficient': 'for z_index!=0: Re 0; Im +1/2',
                               'recipe_id': 'hadron_k3_imag'},
                              {'coefficient': 'for z_index!=0: Re -1/2; Im 0',
                               'recipe_id': 'vacuum_k0_real'},
                              {'coefficient': 'for z_index!=0: Re 0; Im -1/2',
                               'recipe_id': 'vacuum_k0_imag'},
                              {'coefficient': 'for z_index!=0: Re 0; Im -1/2',
                               'recipe_id': 'vacuum_k1_real'},
                              {'coefficient': 'for z_index!=0: Re +1/2; Im 0',
                               'recipe_id': 'vacuum_k1_imag'},
                              {'coefficient': 'for z_index!=0: Re 0; Im +1/2',
                               'recipe_id': 'vacuum_k2_real'},
                              {'coefficient': 'for z_index!=0: Re -1/2; Im 0',
                               'recipe_id': 'vacuum_k2_imag'},
                              {'coefficient': 'for z_index!=0: Re -1/2; Im 0',
                               'recipe_id': 'vacuum_k3_real'},
                              {'coefficient': 'for z_index!=0: Re 0; Im -1/2',
                               'recipe_id': 'vacuum_k3_imag'},
                              {'coefficient': 'for z_index=0: Re -1; Im 0',
                               'recipe_id': 'hadron_z0_Z5'},
                              {'coefficient': 'for z_index=0: Re +1; Im 0',
                               'recipe_id': 'vacuum_z0_Z5'}]},
 'schema_version': '1.1',
 'state_labels': ['hadron', 'vacuum'],
 'state_preparation_interface': {'required_operations': ['prepare_hadron', 'prepare_vacuum'],
                                 'treat_as_black_box': True},
 'time_evolution': {'method': 'charge-preserving even-odd grouped second-order Suzuki product '
                              'formula with exact fixed-width local-block synthesis',
                    'recommended_test_steps': [1, 2, 4, 8, 16],
                    'required': True,
                    'step_operations': [{'angle': 'tau=(z_index*lattice_spacing)/(2*N_T)',
                                         'gate': 'exact_local_block_exponential',
                                         'operation': 'parallel_local_exponentials',
                                         'targets': ['onsite blocks s=0,1,2,3,4']},
                                        {'angle': 'tau=(z_index*lattice_spacing)/(2*N_T)',
                                         'gate': 'exact_local_block_exponential',
                                         'operation': 'parallel_local_exponentials',
                                         'targets': ['link blocks l=0,2']},
                                        {'angle': 'tau=(z_index*lattice_spacing)/N_T',
                                         'gate': 'exact_local_block_exponential',
                                         'operation': 'parallel_local_exponentials',
                                         'targets': ['link blocks l=1,3']},
                                        {'angle': 'tau=(z_index*lattice_spacing)/(2*N_T)',
                                         'gate': 'exact_local_block_exponential',
                                         'operation': 'parallel_local_exponentials',
                                         'targets': ['link blocks l=0,2']},
                                        {'angle': 'tau=(z_index*lattice_spacing)/(2*N_T)',
                                         'gate': 'exact_local_block_exponential',
                                         'operation': 'parallel_local_exponentials',
                                         'targets': ['onsite blocks s=0,1,2,3,4']}],
                    'step_symbol': 'N_T',
                    'symmetry_preservation_rule': 'For z_index!=0 require positive integer '
                                                  'N_T. Every onsite and link block commutes '
                                                  'with Q_qc, so each completed local '
                                                  'exponential and every finite-N_T product '
                                                  'preserve the mandatory Q_qc=0 and Q_qc=-1 '
                                                  'sectors and fermion parity exactly. No '
                                                  'gauge sector exists. The even/odd ordering '
                                                  'can break spatial reflection and '
                                                  'antiunitary lattice symmetries at finite '
                                                  'N_T; compare their diagnostics across '
                                                  'increasing N_T and require convergence. '
                                                  'Broken translations and continuum-only '
                                                  'Lorentz boosts are not imposed. For '
                                                  'z_index=0 accept trotter_steps=0 and build '
                                                  'no evolution.'}}


OBSERVABLE_SPECIFICATION = {'coordinate_convention': {'mode': 'lattice_indices',
                           'spacing_parameter': 'lattice_spacing_a',
                           'temporal_spacing': 'same_as_spatial',
                           'units': 'natural_units_c_1'},
 'expression': '\\bar{\\psi}(z n) \\gamma^- \\psi(0), n^mu=(1,1), \\gamma^- = \\gamma^0 - '
               '\\gamma^1',
 'insertions': [{'id': 'bar_psi_zn',
                 'operator': '\\bar{\\psi}',
                 'position': {'t_index': 'z_index', 'x_index': 'z_index'}},
                {'id': 'psi_0', 'operator': '\\psi', 'position': {'t_index': 0, 'x_index': 0}}],
 'matrix_elements': [{'bra': 'hadron',
                      'coefficient': 1.0,
                      'id': 'hadron_term',
                      'ket': 'hadron'},
                     {'bra': 'vacuum',
                      'coefficient': -1.0,
                      'id': 'vacuum_subtraction',
                      'ket': 'vacuum'}],
 'name': 'user_observable',
 'observable_schema_version': '1.0',
 'ordering': 'as_written',
 'requested_components': ['real', 'imag'],
 'sampling': {'maximum_points': 10000,
              'mode': 'cartesian_product',
              'variables': {'z_index': [-2, -1, 0, 1, 2]}}}


_SYSTEM_QUBITS = 10
_ANCILLA_QUBITS = 1
_SUPPORTED_Z = (-2, -1, 0, 1, 2)
_STATE_LABELS = ("hadron", "vacuum")
_DIRECT_READOUT = "IIIIZIIIII"
_ANCILLA_Z_READOUT = "ZIIIIIIIIII"

_REQUIRED_OPERATOR_RECIPE_IDS = frozenset(
    {
        "endpoint_A_from_z_and_trajectory",
        "endpoint_B_from_trajectory",
        "controlled_pauli_factorwise",
        "onsite_block_h_on_s",
        "link_block_h_link_l",
        "exact_local_block_exponential",
        "same_state_interferometer",
        "coincident_Z5_direct",
    }
)
_EMBEDDED_OPERATOR_RECIPE_IDS = frozenset(
    entry["id"] for entry in OBSERVABLE_CONTRACT["operator_recipes"]
)
if not _REQUIRED_OPERATOR_RECIPE_IDS.issubset(_EMBEDDED_OPERATOR_RECIPE_IDS):
    raise RuntimeError("Embedded observable contract lacks a required operator recipe")

_CIRCUIT_RECIPE_BINDINGS = {}
for _entry in OBSERVABLE_CONTRACT["measurement"]["circuit_recipes"]:
    _operation = _entry["operations"][0]
    _CIRCUIT_RECIPE_BINDINGS[_entry["recipe_id"]] = (
        _entry["component"],
        _operation["reference"],
        dict(_operation["bindings"]),
    )

_EXPECTED_RECIPE_IDS = tuple(
    OBSERVABLE_CONTRACT["measurement"]["circuit_recipe_ids"]
)
if set(_CIRCUIT_RECIPE_BINDINGS) != set(_EXPECTED_RECIPE_IDS):
    raise RuntimeError("Embedded circuit recipe definitions are inconsistent")

_PAULI_I = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MATRICES = {
    "I": _PAULI_I,
    "X": _PAULI_X,
    "Y": _PAULI_Y,
    "Z": _PAULI_Z,
}

_MASS = 0.0
_COUPLING = 0.0
_LATTICE_SPACING = 1.0
_WILSON_PARAMETER = 0.0
_ONSITE_XX_YY_COEFFICIENT = 0.0
_ONSITE_ZZ_COEFFICIENT = 0.0
_LINK_DERIVATIVE_COEFFICIENT = 0.0
_LINK_WILSON_COEFFICIENT = 0.0
_MATRIX_ELEMENT_SIGNS = {"hadron": 1.0, "vacuum": -1.0}
_RECONSTRUCTION_HALF = 0.5
_SYNTHESIS_CACHE = {}


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _exact_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _parameter_values(parameter_contract):
    if not isinstance(parameter_contract, dict):
        raise TypeError("parameter_contract must be a dict")
    required_top = {
        "schema_version",
        "model",
        "parameters",
        "constraints",
        "discretizations",
        "external_states",
        "validation_obligations",
    }
    if not required_top.issubset(parameter_contract):
        raise ValueError("parameter_contract is missing required top-level content")

    entries = parameter_contract["parameters"]
    if not isinstance(entries, list):
        raise ValueError("parameter_contract['parameters'] must be a list")

    values = {}
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "selected_value" not in entry:
            raise ValueError("Every parameter entry must contain name and selected_value")
        name = entry["name"]
        if name in values:
            raise ValueError(f"Duplicate parameter name: {name!r}")
        values[name] = entry["selected_value"]
    return values


def _validate_observable_specification(observable_specification):
    if not isinstance(observable_specification, dict):
        raise TypeError("observable_specification must be a dict")

    required_top = {
        "observable_schema_version",
        "name",
        "expression",
        "ordering",
        "coordinate_convention",
        "insertions",
        "matrix_elements",
        "requested_components",
        "sampling",
    }
    if not required_top.issubset(observable_specification):
        raise ValueError("observable_specification is missing required top-level content")

    if observable_specification["name"] != "user_observable":
        raise ValueError("Unsupported observable name")
    if observable_specification["ordering"] != "as_written":
        raise ValueError("The observable must retain as-written ordering")
    if observable_specification["requested_components"] != ["real", "imag"]:
        raise ValueError("The observable must request real and imaginary components")

    convention = observable_specification["coordinate_convention"]
    expected_convention = {
        "mode": "lattice_indices",
        "spacing_parameter": "lattice_spacing_a",
        "temporal_spacing": "same_as_spatial",
        "units": "natural_units_c_1",
    }
    if convention != expected_convention:
        raise ValueError("Unsupported observable coordinate convention")

    insertions = observable_specification["insertions"]
    if not isinstance(insertions, list) or len(insertions) != 2:
        raise ValueError("Exactly two ordered observable insertions are required")
    if insertions[0] != {
        "id": "bar_psi_zn",
        "operator": "\\bar{\\psi}",
        "position": {"t_index": "z_index", "x_index": "z_index"},
    }:
        raise ValueError("The first insertion does not match bar_psi_zn")
    if insertions[1] != {
        "id": "psi_0",
        "operator": "\\psi",
        "position": {"t_index": 0, "x_index": 0},
    }:
        raise ValueError("The second insertion does not match psi_0")

    matrix_elements = observable_specification["matrix_elements"]
    if not isinstance(matrix_elements, list) or len(matrix_elements) != 2:
        raise ValueError("Exactly two matrix-element terms are required")

    observed_terms = {}
    for term in matrix_elements:
        bra = term.get("bra")
        ket = term.get("ket")
        if bra != ket or bra not in _STATE_LABELS:
            raise ValueError("Only the declared diagonal external-state terms are supported")
        observed_terms[bra] = _finite_real(
            term.get("coefficient"), f"{bra} matrix-element coefficient"
        )
    if observed_terms != {"hadron": 1.0, "vacuum": -1.0}:
        raise ValueError("Observable matrix-element coefficients do not match the protocol")

    sampling = observable_specification["sampling"]
    if sampling.get("mode") != "cartesian_product":
        raise ValueError("Unsupported sampling mode")
    variables = sampling.get("variables")
    if variables != {"z_index": list(_SUPPORTED_Z)}:
        raise ValueError("The z_index sampling domain must be [-2,-1,0,1,2]")


def configure_runtime(parameter_contract, observable_specification):
    global _MASS
    global _COUPLING
    global _LATTICE_SPACING
    global _WILSON_PARAMETER
    global _ONSITE_XX_YY_COEFFICIENT
    global _ONSITE_ZZ_COEFFICIENT
    global _LINK_DERIVATIVE_COEFFICIENT
    global _LINK_WILSON_COEFFICIENT
    global _MATRIX_ELEMENT_SIGNS
    global _RECONSTRUCTION_HALF
    global _SYNTHESIS_CACHE

    values = _parameter_values(parameter_contract)
    required_parameters = {
        "bare_quark_mass",
        "bare_four_fermion_coupling",
        "lattice_spacing",
        "spatial_site_count",
        "wilson_parameter",
        "fermionic_mode_count",
        "bare coupling: g",
        "bare quark mass: m",
        "lattice spacing: a",
        "number of lattice site: L",
    }
    if not required_parameters.issubset(values):
        missing = sorted(required_parameters.difference(values))
        raise ValueError(f"Missing required parameters: {missing}")

    model = parameter_contract["model"]
    if model != {
        "gauge_structure": "none",
        "name": "one flavor NJL model",
        "spacetime_dimension": "1+1",
    }:
        raise ValueError("Unsupported model structure")

    mass = _finite_real(values["bare_quark_mass"], "bare_quark_mass")
    coupling = _finite_real(
        values["bare_four_fermion_coupling"],
        "bare_four_fermion_coupling",
    )
    spacing = _finite_real(values["lattice_spacing"], "lattice_spacing")
    wilson = _finite_real(values["wilson_parameter"], "wilson_parameter")
    site_count = _exact_integer(values["spatial_site_count"], "spatial_site_count")
    mode_count = _exact_integer(values["fermionic_mode_count"], "fermionic_mode_count")

    if spacing != 1.0:
        raise ValueError("The immutable lattice_spacing must equal 1")
    if site_count != 5:
        raise ValueError("The immutable spatial_site_count must equal 5")
    if mode_count != 2 * site_count or mode_count != _SYSTEM_QUBITS:
        raise ValueError("fermionic_mode_count must equal 10 and 2*spatial_site_count")
    if wilson < 0.0:
        raise ValueError("wilson_parameter must be nonnegative")

    duplicate_mass = _finite_real(values["bare quark mass: m"], "bare quark mass: m")
    duplicate_coupling = _finite_real(values["bare coupling: g"], "bare coupling: g")
    duplicate_spacing = _finite_real(values["lattice spacing: a"], "lattice spacing: a")
    duplicate_sites = _exact_integer(
        values["number of lattice site: L"], "number of lattice site: L"
    )
    if duplicate_mass != mass:
        raise ValueError("The two authoritative bare-mass values disagree")
    if duplicate_coupling != coupling:
        raise ValueError("The two authoritative coupling values disagree")
    if duplicate_spacing != spacing:
        raise ValueError("The two authoritative lattice-spacing values disagree")
    if duplicate_sites != site_count:
        raise ValueError("The two authoritative site-count values disagree")

    states = parameter_contract["external_states"]
    if states.get("labels") != ["hadron", "vacuum"]:
        raise ValueError("External-state labels must be ['hadron', 'vacuum']")
    definitions = states.get("definitions")
    if not isinstance(definitions, dict) or set(definitions) != set(_STATE_LABELS):
        raise ValueError("External-state definitions do not match the protocol")
    if "level_index=1" not in definitions["hadron"]:
        raise ValueError("hadron must be charge-zero level_index 1")
    if "level_index=0" not in definitions["vacuum"]:
        raise ValueError("vacuum must be charge-zero level_index 0")

    discretizations = parameter_contract["discretizations"]
    if not isinstance(discretizations, list) or len(discretizations) != 2:
        raise ValueError("Exactly two authoritative discretizations are required")
    by_sector = {entry.get("sector"): entry for entry in discretizations}
    if set(by_sector) != {"spacetime", "fermion"}:
        raise ValueError("Missing spacetime or fermion discretization")

    spacetime = by_sector["spacetime"]["settings"]
    if spacetime.get("spatial_site_indices") != [-2, -1, 0, 1, 2]:
        raise ValueError("Spatial site indices do not match the five-site lattice")
    if spacetime.get("spatial_boundary_condition") != "open":
        raise ValueError("Only open spatial boundaries are supported")
    if spacetime.get("nearest_neighbor_link_indices") != [
        [-2, -1],
        [-1, 0],
        [0, 1],
        [1, 2],
    ]:
        raise ValueError("Nearest-neighbor links do not match the protocol")

    fermion = by_sector["fermion"]["settings"]
    if fermion.get("components_per_site") != 2:
        raise ValueError("Exactly two fermion components per site are required")
    if fermion.get("fermionic_mode_count") != _SYSTEM_QUBITS:
        raise ValueError("The fermion discretization must contain ten modes")
    if fermion.get("qubit_count") != _SYSTEM_QUBITS:
        raise ValueError("The fermion discretization must contain ten qubits")
    if fermion.get("charge_zero_sector_dimension") != 252:
        raise ValueError("The charge-zero sector dimension must equal 252")
    if fermion.get("intermediate_charge_minus_one_sector_dimension") != 210:
        raise ValueError("The charge-minus-one sector dimension must equal 210")

    _validate_observable_specification(observable_specification)

    _MASS = mass
    _COUPLING = coupling
    _LATTICE_SPACING = spacing
    _WILSON_PARAMETER = wilson
    _ONSITE_XX_YY_COEFFICIENT = (
        _MASS + _WILSON_PARAMETER / _LATTICE_SPACING
    ) / 2.0
    _ONSITE_ZZ_COEFFICIENT = _COUPLING / 2.0
    _LINK_DERIVATIVE_COEFFICIENT = 1.0 / (4.0 * _LATTICE_SPACING)
    _LINK_WILSON_COEFFICIENT = -_WILSON_PARAMETER / (
        4.0 * _LATTICE_SPACING
    )

    matrix_elements = observable_specification["matrix_elements"]
    _MATRIX_ELEMENT_SIGNS = {
        term["bra"]: _finite_real(
            term["coefficient"], f"{term['bra']} matrix-element coefficient"
        )
        for term in matrix_elements
    }
    _RECONSTRUCTION_HALF = 1.0 / 2.0
    _SYNTHESIS_CACHE = {}


def _sample_z(sample_values):
    if not isinstance(sample_values, dict) or set(sample_values) != {"z_index"}:
        raise ValueError("sample_values must contain exactly one key: 'z_index'")
    z_index = sample_values["z_index"]
    if isinstance(z_index, bool) or not isinstance(z_index, (int, np.integer)):
        raise ValueError("z_index must be an integer")
    z_index = int(z_index)
    if z_index not in _SUPPORTED_Z:
        raise ValueError("z_index must be one of -2,-1,0,1,2")
    return z_index


def _validate_trotter_steps(trotter_steps, z_index):
    if isinstance(trotter_steps, bool) or not isinstance(
        trotter_steps, (int, np.integer)
    ):
        raise ValueError("trotter_steps must be an integer")
    trotter_steps = int(trotter_steps)
    if z_index == 0:
        if trotter_steps < 0:
            raise ValueError("trotter_steps cannot be negative")
    elif trotter_steps <= 0:
        raise ValueError("Nonzero-time recipes require positive trotter_steps")
    return trotter_steps


def _validate_state_circuits(state_circuits):
    if not isinstance(state_circuits, dict) or set(state_circuits) != set(_STATE_LABELS):
        raise ValueError("state_circuits must map exactly 'hadron' and 'vacuum'")

    forbidden = {"measure", "reset", "initialize"}
    validated = {}
    for state in _STATE_LABELS:
        circuit = state_circuits[state]
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError(f"state_circuits[{state!r}] must be a QuantumCircuit")
        if circuit.num_qubits != _SYSTEM_QUBITS:
            raise ValueError(f"{state} state circuit must have ten qubits")
        if circuit.num_clbits != 0:
            raise ValueError(f"{state} state circuit must have no classical bits")
        if circuit.parameters:
            raise ValueError(f"{state} state circuit must be fully parameter-bound")
        for instruction in circuit.data:
            if instruction.operation.name in forbidden:
                raise ValueError(
                    f"{state} state circuit contains forbidden operation "
                    f"{instruction.operation.name!r}"
                )
        validated[state] = circuit
    return validated


def _pauli_matrix(label):
    result = np.array([[1.0 + 0.0j]])
    for character in label:
        try:
            factor = _PAULI_MATRICES[character]
        except KeyError as exc:
            raise ValueError(f"Invalid Pauli label {label!r}") from exc
        result = np.kron(result, factor)
    return result


def _onsite_generator():
    return (
        _ONSITE_XX_YY_COEFFICIENT
        * (_pauli_matrix("XX") + _pauli_matrix("YY"))
        + _ONSITE_ZZ_COEFFICIENT * _pauli_matrix("ZZ")
    )


def _link_generator():
    derivative = (
        _pauli_matrix("IYZX")
        - _pauli_matrix("IXZY")
        - _pauli_matrix("YZXI")
        + _pauli_matrix("XZYI")
    )
    wilson = (
        _pauli_matrix("XZZX")
        + _pauli_matrix("YZZY")
        + _pauli_matrix("IXXI")
        + _pauli_matrix("IYYI")
    )
    return (
        _LINK_DERIVATIVE_COEFFICIENT * derivative
        + _LINK_WILSON_COEFFICIENT * wilson
    )


def _synthesized_local_exponential(block_kind, tau):
    key = (
        block_kind,
        float(tau),
        _MASS,
        _COUPLING,
        _LATTICE_SPACING,
        _WILSON_PARAMETER,
    )
    cached = _SYNTHESIS_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    if block_kind == "onsite":
        generator = _onsite_generator()
        width = 2
    elif block_kind == "link":
        generator = _link_generator()
        width = 4
    else:
        raise ValueError(f"Unknown local block kind: {block_kind!r}")

    unitary = expm(-1j * float(tau) * generator)
    synthesized = qs_decomposition(unitary)
    if synthesized.num_qubits != width:
        raise RuntimeError("Exact local synthesis returned an incorrect width")
    if synthesized.parameters:
        raise RuntimeError("Exact local synthesis left unbound parameters")
    if synthesized.num_clbits:
        raise RuntimeError("Exact local synthesis introduced classical bits")

    _SYNTHESIS_CACHE[key] = synthesized.copy()
    return synthesized


def _append_local_layer(circuit, block_kind, indices, tau):
    block = _synthesized_local_exponential(block_kind, tau)
    if block_kind == "onsite":
        for index in indices:
            circuit.compose(block, qubits=[2 * index, 2 * index + 1], inplace=True)
    else:
        for index in indices:
            base = 2 * index
            circuit.compose(
                block,
                qubits=[base, base + 1, base + 2, base + 3],
                inplace=True,
            )


def _append_s2_evolution(circuit, total_time, trotter_steps):
    half_tau = float(total_time) / (2.0 * trotter_steps)
    full_tau = float(total_time) / trotter_steps
    for _ in range(trotter_steps):
        _append_local_layer(circuit, "onsite", (0, 1, 2, 3, 4), half_tau)
        _append_local_layer(circuit, "link", (0, 2), half_tau)
        _append_local_layer(circuit, "link", (1, 3), full_tau)
        _append_local_layer(circuit, "link", (0, 2), half_tau)
        _append_local_layer(circuit, "onsite", (0, 1, 2, 3, 4), half_tau)


def _endpoint_a(z_index, trajectory):
    use_x = trajectory in (0, 1)
    factor = "X" if use_x else "Y"
    if z_index == -2:
        return "IIIIIIII" + factor + "Z"
    if z_index == -1:
        return "IIIIII" + factor + "ZZZ"
    if z_index == 1:
        return "II" + factor + "ZZZZZZZ"
    if z_index == 2:
        return factor + "ZZZZZZZZZ"
    raise ValueError("Endpoint A is undefined at z_index=0")


def _endpoint_b(trajectory):
    factor = "X" if trajectory in (0, 2) else "Y"
    return "IIII" + factor + "ZZZZZ"


def _append_controlled_pauli(circuit, ancilla, pauli_label):
    if len(pauli_label) != _SYSTEM_QUBITS:
        raise ValueError("Controlled Pauli must have a ten-qubit label")
    for qubit, factor in enumerate(reversed(pauli_label)):
        if factor == "I":
            continue
        if factor == "X":
            circuit.cx(ancilla, qubit)
        elif factor == "Y":
            circuit.sdg(qubit)
            circuit.cx(ancilla, qubit)
            circuit.s(qubit)
        elif factor == "Z":
            circuit.cz(ancilla, qubit)
        else:
            raise ValueError(f"Invalid Pauli factor: {factor!r}")


def _same_state_interferometer(
    state_circuit,
    z_index,
    trajectory,
    component,
    trotter_steps,
):
    circuit = QuantumCircuit(_SYSTEM_QUBITS + _ANCILLA_QUBITS)
    circuit.compose(state_circuit, qubits=range(_SYSTEM_QUBITS), inplace=True)

    ancilla = _SYSTEM_QUBITS
    circuit.h(ancilla)
    _append_controlled_pauli(circuit, ancilla, _endpoint_b(trajectory))
    _append_s2_evolution(
        circuit,
        total_time=z_index * _LATTICE_SPACING,
        trotter_steps=trotter_steps,
    )
    _append_controlled_pauli(
        circuit,
        ancilla,
        _endpoint_a(z_index, trajectory),
    )

    if component == "real":
        circuit.h(ancilla)
    elif component == "imag":
        circuit.sdg(ancilla)
        circuit.h(ancilla)
    else:
        raise ValueError(f"Unsupported component: {component!r}")

    if circuit.parameters:
        raise RuntimeError("Generated interferometer contains unbound parameters")
    if circuit.num_clbits:
        raise RuntimeError("Generated interferometer contains classical bits")
    return circuit


def _direct_z5_circuit(state_circuit):
    circuit = QuantumCircuit(_SYSTEM_QUBITS)
    circuit.compose(state_circuit, qubits=range(_SYSTEM_QUBITS), inplace=True)
    if circuit.parameters:
        raise RuntimeError("Generated direct circuit contains unbound parameters")
    if circuit.num_clbits:
        raise RuntimeError("Generated direct circuit contains classical bits")
    return circuit


def build_observable_measurements(
    state_circuits,
    trotter_steps,
    sample_values,
):
    z_index = _sample_z(sample_values)
    trotter_steps = _validate_trotter_steps(trotter_steps, z_index)
    states = _validate_state_circuits(state_circuits)

    recipes = []
    if z_index == 0:
        for recipe_id in ("hadron_z0_Z5", "vacuum_z0_Z5"):
            component, reference, bindings = _CIRCUIT_RECIPE_BINDINGS[recipe_id]
            if reference != "coincident_Z5_direct":
                raise RuntimeError("Direct recipe does not resolve to coincident_Z5_direct")
            state = bindings["state"]
            recipes.append(
                {
                    "recipe_id": recipe_id,
                    "component": component,
                    "circuit": _direct_z5_circuit(states[state]),
                    "readout_pauli": _DIRECT_READOUT,
                }
            )
        return recipes

    for recipe_id in _EXPECTED_RECIPE_IDS:
        component, reference, bindings = _CIRCUIT_RECIPE_BINDINGS[recipe_id]
        if reference != "same_state_interferometer":
            continue
        state = bindings["state"]
        trajectory = bindings["trajectory"]
        recipes.append(
            {
                "recipe_id": recipe_id,
                "component": component,
                "circuit": _same_state_interferometer(
                    states[state],
                    z_index,
                    trajectory,
                    component,
                    trotter_steps,
                ),
                "readout_pauli": _ANCILLA_Z_READOUT,
            }
        )
    return recipes


def _expectation_value(expectations, recipe_id):
    if recipe_id not in expectations:
        raise ValueError(f"Missing expectation for recipe {recipe_id!r}")
    value = expectations[recipe_id]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"Expectation {recipe_id!r} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Expectation {recipe_id!r} must be finite")
    return value


def reconstruct_observable(expectations, sample_values):
    if not isinstance(expectations, dict):
        raise TypeError("expectations must be a dict")
    z_index = _sample_z(sample_values)

    if z_index == 0:
        required = {"hadron_z0_Z5", "vacuum_z0_Z5"}
        if set(expectations) != required:
            raise ValueError(
                "z_index=0 expectations must contain exactly the two direct recipes"
            )
        real = (
            _expectation_value(expectations, "vacuum_z0_Z5")
            - _expectation_value(expectations, "hadron_z0_Z5")
        )
        return complex(real, 0.0)

    required = {
        f"{state}_k{k}_{component}"
        for state in _STATE_LABELS
        for k in range(4)
        for component in ("real", "imag")
    }
    if set(expectations) != required:
        missing = sorted(required.difference(expectations))
        extra = sorted(set(expectations).difference(required))
        raise ValueError(
            f"Nonzero-time expectation keys mismatch; missing={missing}, extra={extra}"
        )

    total_real = 0.0
    total_imag = 0.0
    for state in _STATE_LABELS:
        r0 = _expectation_value(expectations, f"{state}_k0_real")
        i0 = _expectation_value(expectations, f"{state}_k0_imag")
        r1 = _expectation_value(expectations, f"{state}_k1_real")
        i1 = _expectation_value(expectations, f"{state}_k1_imag")
        r2 = _expectation_value(expectations, f"{state}_k2_real")
        i2 = _expectation_value(expectations, f"{state}_k2_imag")
        r3 = _expectation_value(expectations, f"{state}_k3_real")
        i3 = _expectation_value(expectations, f"{state}_k3_imag")

        state_sign = _MATRIX_ELEMENT_SIGNS[state]
        total_real += state_sign * _RECONSTRUCTION_HALF * (
            r0 - i1 + i2 + r3
        )
        total_imag += state_sign * _RECONSTRUCTION_HALF * (
            i0 + r1 - r2 + i3
        )

    return complex(total_real, total_imag)


configure_runtime(PARAMETER_CONTRACT, OBSERVABLE_SPECIFICATION)


