import json
import math
from numbers import Real

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliGate


PARAMETER_CONTRACT = {'constraints': [{'applies_when': 'always',
                  'id': 'equal_temporal_spatial_spacing',
                  'parameters': ['temporal_lattice_spacing', 'lattice_spacing'],
                  'reason': 'The continuum contract requires equal temporal and spatial '
                            'spacing.',
                  'requirement': 'temporal_lattice_spacing must equal lattice_spacing',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'open_boundary_gauss_reconstruction',
                  'parameters': ['bare_gauge_coupling'],
                  'reason': 'The ordered fermion insertions visit an intermediate '
                            'charge-minus-one sector.',
                  'requirement': 'Use the declared reflection-symmetric open-boundary '
                                 'electric-field reconstruction in every total-charge sector '
                                 'visited by the ordered observable.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'preserve_unshifted_hamiltonian',
                  'parameters': [],
                  'reason': 'The requested lattice Hamiltonian is the direct unshifted '
                            'discretization.',
                  'requirement': 'Do not subtract any vacuum or target-state energy and do not '
                                 'introduce state-dependent Hamiltonian coefficients.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'fixed_gamma_convention',
                  'parameters': ['lightcone_normalization'],
                  'reason': 'These continuum conventions are immutable.',
                  'requirement': 'Use gamma_0=sigma_1, gamma_1=-i*sigma_2, alpha=sigma_3, '
                                 'beta=sigma_1 and gamma_minus=gamma_0-gamma_1 with '
                                 'normalization one.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'observable_range_fit',
                  'parameters': ['lattice_spacing'],
                  'reason': 'No requested observable sample may be dropped.',
                  'requirement': 'The site map s(z)=z+2 must be valid for every supplied z, '
                                 'including both endpoints.',
                  'severity': 'fatal'},
                 {'applies_when': 'always',
                  'id': 'no_gauge_flux_truncation',
                  'parameters': [],
                  'reason': 'Reintroducing links would change the selected finite '
                            'representation.',
                  'requirement': 'Do not introduce independent gauge-link variables or an '
                                 'electric-flux cutoff after exact Gauss-law elimination.',
                  'severity': 'fatal'}],
 'discretizations': [{'parameter_references': ['lattice_spacing', 'temporal_lattice_spacing'],
                      'scheme': 'centered one-dimensional spatial lattice with equal temporal '
                                'spacing',
                      'sector': 'spacetime',
                      'settings': {'gauss_relation_count': 5,
                                   'observable_site_map': 's(z)=z+2',
                                   'plaquette_count': 0,
                                   'spatial_boundary_condition': 'open',
                                   'spatial_coordinate': 'x_N=a*(N-2)',
                                   'spatial_dimension': 1,
                                   'spatial_link_count': 4,
                                   'spatial_link_index_range': 'ell=0,...,3',
                                   'spatial_site_count': 5,
                                   'spatial_site_index_range': 'N=0,...,4',
                                   'temporal_spacing_relation': 'a_t=a'},
                      'truncation': {'reason': 'The immutable five-site lattice supports every '
                                               'requested separation.',
                                     'scheme': 'finite spatial volume',
                                     'settings': {'coordinate_interval': '[-2a,2a]'}}},
                     {'parameter_references': ['bare_fermion_mass',
                                               'wilson_parameter',
                                               'charge_reference_occupancy'],
                      'scheme': 'two-component Wilson fermions',
                      'sector': 'fermion',
                      'settings': {'alpha': 'sigma_3',
                                   'beta': 'sigma_1',
                                   'charge_density': 'q_N=sum_alpha '
                                                     'c_dagger[N,alpha]*c[N,alpha]-1',
                                   'component_count_per_site': 2,
                                   'fermionic_mode_count': 10,
                                   'gamma_0': 'sigma_1',
                                   'gamma_1': '-i*sigma_2',
                                   'wilson_parameter': 1},
                      'truncation': {'reason': 'Fermionic occupation is intrinsically finite '
                                               'and is not an approximate field truncation.',
                                     'scheme': 'exact fermionic occupation',
                                     'settings': {'local_mode_occupations': [0, 1],
                                                  'neutral_sector_dimension': 252,
                                                  'total_fock_dimension': 1024}}},
                     {'parameter_references': ['bare_gauge_coupling'],
                      'scheme': 'spatial axial gauge with exact open-boundary Gauss-law '
                                'elimination',
                      'sector': 'gauge',
                      'settings': {'effective_degree_count': 10,
                                   'effective_qubit_count': 10,
                                   'electric_field_reconstruction': 'E_-1=-(g/2)*Q_lat; '
                                                                    'E_ell=g*(S_ell-Q_lat/2) '
                                                                    'for ell=0,...,3; '
                                                                    'E_4=(g/2)*Q_lat',
                                   'external_neutral_sector_dimension': 252,
                                   'gauge_condition': 'A_1=0',
                                   'gauss_identity': 'E_N-E_{N-1}=g*q_N at every lattice '
                                                     'vertex',
                                   'independent_constraint_count': 4,
                                   'independent_gauge_degree_count': 0,
                                   'kinematic_dimension_after_reduction': 1024,
                                   'locality_change': 'Local link-electric energy becomes a '
                                                      'long-range fermionic Coulomb '
                                                      'interaction.',
                                   'partial_charge': 'S_ell=sum_{N=0}^{ell} q_N',
                                   'physical_dimension_with_all_required_charge_sectors': 1024,
                                   'scalar_potential_reconstruction': 'A0_N=a*g*sum_{ell=0}^{3}(indicator(N<=ell)-1/2)*(S_ell-Q_lat/2)',
                                   'total_charge': 'Q_lat=sum_{N=0}^{4} q_N'},
                      'truncation': {'reason': 'No independent infinite-dimensional gauge '
                                               'Hilbert space remains after exact Gauss-law '
                                               'elimination.',
                                     'scheme': 'none',
                                     'settings': {'electric_flux_cutoff': 'none',
                                                  'gauge_links': 'exactly eliminated'}}}],
 'external_states': {'definitions': {'phi': 'Quantum-number-sector eigenstate; level_index=0; '
                                            'quantum_numbers={"charge": 0}',
                                     'psi': 'Quantum-number-sector eigenstate; level_index=1; '
                                            'quantum_numbers={"charge": 0}'},
                     'equivalent_groups': [],
                     'labels': ['phi', 'psi']},
 'model': {'gauge_structure': 'local U(1)',
           'name': 'Schwinger model',
           'spacetime_dimension': '1+1'},
 'parameters': [{'domain': {'minimum': 0},
                 'name': 'bare_gauge_coupling',
                 'required_for_run': True,
                 'role': 'Bare U(1) gauge coupling.',
                 'selected_value': 0.5,
                 'symbol': 'g',
                 'type': 'real',
                 'unit': 'inverse_length'},
                {'domain': {'rule': 'finite real number'},
                 'name': 'bare_fermion_mass',
                 'required_for_run': True,
                 'role': 'Bare Dirac fermion mass.',
                 'selected_value': 0.5,
                 'symbol': 'm',
                 'type': 'real',
                 'unit': 'inverse_length'},
                {'domain': {'exclusive_minimum': 0},
                 'name': 'lattice_spacing',
                 'required_for_run': True,
                 'role': 'Spatial lattice spacing.',
                 'selected_value': 1.0,
                 'symbol': 'a',
                 'type': 'real',
                 'unit': 'length'},
                {'domain': {'exclusive_minimum': 0},
                 'name': 'temporal_lattice_spacing',
                 'required_for_run': True,
                 'role': 'Temporal spacing, fixed equal to the spatial spacing.',
                 'selected_value': 1.0,
                 'symbol': 'a_t',
                 'type': 'real',
                 'unit': 'time'},
                {'domain': {'rule': 'nonzero finite real number'},
                 'name': 'wilson_parameter',
                 'required_for_run': True,
                 'role': 'Wilson term coefficient used to remove fermion doublers.',
                 'selected_value': 1.0,
                 'symbol': 'r',
                 'type': 'real',
                 'unit': None},
                {'domain': {'allowed_values': [1]},
                 'name': 'charge_reference_occupancy',
                 'required_for_run': True,
                 'role': 'Fixed state-independent occupancy reference in q_N=c_N^dagger '
                         'c_N-n_ref.',
                 'selected_value': 1,
                 'symbol': 'n_ref',
                 'type': 'integer',
                 'unit': None},
                {'domain': {'allowed_values': [1.0]},
                 'name': 'lightcone_normalization',
                 'required_for_run': True,
                 'role': 'Normalization in gamma^-=gamma^0-gamma^1 and X(z)=az(1,1).',
                 'selected_value': 1.0,
                 'symbol': 'c_lc',
                 'type': 'real',
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
                 'selected_value': 1.0,
                 'source_ref': '/user_input/parameters/lattice spacing: a',
                 'symbol': 'lattice spacing: a',
                 'type': 'real',
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
 'validation_obligations': [{'description': 'Confirm that phi is the zero-based level 0 '
                                            'eigenstate and psi is the zero-based level 1 '
                                            'eigenstate of H_lat within the Q_lat=0 sector, '
                                            'without identifying the two states.',
                             'id': 'external_state_sector_and_level_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that E_N-E_{N-1}-g*q_N vanishes as an '
                                            'operator identity at every lattice vertex under '
                                            'the declared reconstruction, including the '
                                            'intermediate charge-minus-one sector.',
                             'id': 'physical_gauss_sector_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm that [H_lat,Q_lat]=0 in the full '
                                            '1024-dimensional reduced Fock space.',
                             'id': 'charge_commutator_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'},
                            {'description': 'Confirm the ordered path Q=0 -> Q=-1 -> Q=-1 -> '
                                            'Q=0 for the right-to-left action of psi '
                                            'insertion, Wilson transporter, and bar-psi '
                                            'insertion.',
                             'id': 'ordered_sector_path_check',
                             'severity': 'fatal',
                             'stage': 'pre_simulation'},
                            {'description': 'Confirm that the constructed Hamiltonian contains '
                                            'no additive vacuum-energy subtraction or '
                                            'target-state-dependent coefficient.',
                             'id': 'no_energy_shift_check',
                             'severity': 'fatal',
                             'stage': 'post_construction'}]}

OBSERVABLE_CONTRACT = {'adapter_interface': {'builder': 'build_observable_measurements'},
 'contract_type': 'observable_measurement',
 'literature_sources': [{'applicability': 'modified: supports the fermion-Wilson-line-fermion '
                                          'operator structure, but does not provide the '
                                          'required independent-state transition-amplitude '
                                          'circuit',
                         'locator': 'Section "Ansatz for interacting wave-packet creation '
                                    'operator", Eqs. (ansatz1) through (B-def), including the '
                                    'paragraph defining M_{m,n}',
                         'source_path': 'arXiv-2402.00840v3/arxiv_v3_published.tex'},
                        {'applicability': 'modified: supports finite real-time product-formula '
                                          'circuits; the transition interferometer and '
                                          'reconstruction below are independently derived',
                         'locator': 'Section "Real-time Simulations using IBM\'s Digital '
                                    'Quantum Computers", paragraphs defining physics and '
                                    'forward-backward mitigation circuits',
                         'source_path': 'arXiv-2401.08044v2/main_v6.tex'}],
 'measurement': {'ancilla_qubits': 1,
                 'circuit_recipe_ids': ['sample_z1__XX_real',
                                        'sample_z1__XX_imag',
                                        'sample_z1__XY_real',
                                        'sample_z1__XY_imag',
                                        'sample_z1__YX_real',
                                        'sample_z1__YX_imag',
                                        'sample_z1__YY_real',
                                        'sample_z1__YY_imag'],
                 'circuit_recipes': [{'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_X_b_X',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'H_on_ancilla_then_readout_pauli_Z_ancilla'}],
                                      'recipe_id': 'sample_z1__XX_real'},
                                     {'component': 'imag',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_X_b_X',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'Sdg_then_H_on_ancilla_then_readout_pauli_Z_ancilla_for_positive_imaginary_part'}],
                                      'recipe_id': 'sample_z1__XX_imag'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_X_b_Y',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'H_on_ancilla_then_readout_pauli_Z_ancilla'}],
                                      'recipe_id': 'sample_z1__XY_real'},
                                     {'component': 'imag',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_X_b_Y',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'Sdg_then_H_on_ancilla_then_readout_pauli_Z_ancilla_for_positive_imaginary_part'}],
                                      'recipe_id': 'sample_z1__XY_imag'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_Y_b_X',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'H_on_ancilla_then_readout_pauli_Z_ancilla'}],
                                      'recipe_id': 'sample_z1__YX_real'},
                                     {'component': 'imag',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_Y_b_X',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'Sdg_then_H_on_ancilla_then_readout_pauli_Z_ancilla_for_positive_imaginary_part'}],
                                      'recipe_id': 'sample_z1__YX_imag'},
                                     {'component': 'real',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_Y_b_Y',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'H_on_ancilla_then_readout_pauli_Z_ancilla'}],
                                      'recipe_id': 'sample_z1__YY_real'},
                                     {'component': 'imag',
                                      'operations': [{'operation': 'compose',
                                                      'reference': 'validate_selected_sample'},
                                                     {'operation': 'prepare_interferometer_prefix',
                                                      'reference': 'ancilla_H_then_unconditional_prepare_phi_then_controlled_inverse_phi_then_controlled_prepare_psi'},
                                                     {'operation': 'compose_controlled_trajectory_with_a_Y_b_Y',
                                                      'reference': 'controlled_trajectory_V_ab'},
                                                     {'operation': 'ancilla_basis_and_readout',
                                                      'reference': 'Sdg_then_H_on_ancilla_then_readout_pauli_Z_ancilla_for_positive_imaginary_part'}],
                                      'recipe_id': 'sample_z1__YY_imag'}],
                 'components': ['real', 'imag'],
                 'protocol_id': 'schwinger_10q_independent_state_wilson_transition_interferometer'},
 'operator_recipes': [{'id': 'validate_selected_sample',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'CLASSICAL_VALIDATE',
                                       'operation': 'validate_immutable_assignment',
                                       'targets': ['sample_values.z1 in {-2,-1,0,1,2}',
                                                   'sample_values.z2 equals 0',
                                                   'no missing keys',
                                                   'no extra keys',
                                                   'trotter_steps is positive integer']}]},
                      {'id': 'jw_left_endpoint_pauli',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PAULI_STRING',
                                       'operation': 'indexed_pauli_string',
                                       'targets': ['j=2*(sample_values.z1+2)+1',
                                                   'Z on q[0] through q[j-1]',
                                                   'a on q[j], where a is X or Y']}]},
                      {'id': 'jw_right_origin_pauli',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'PAULI_STRING',
                                       'operation': 'indexed_pauli_string',
                                       'targets': ['Z on q[0] through q[4]',
                                                   'b on q[5], where b is X or Y']}]},
                      {'id': 'pauli_rotation_uncontrolled',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'H for X; Sdg then H for Y; I for Z',
                                       'operation': 'basis_change_for_each_nonidentity_factor',
                                       'targets': ['support(P)']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'CNOT_LADDER',
                                       'operation': 'parity_compute',
                                       'targets': ['ordered support(P)']},
                                      {'angle': '2*theta for exp(-i*theta*P)',
                                       'controls': [],
                                       'gate': 'RZ',
                                       'operation': 'central_rotation',
                                       'targets': ['last qubit in ordered support(P)']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'INVERSE_OF_PREVIOUS_CLIFFORDS',
                                       'operation': 'parity_uncompute_and_inverse_basis_change',
                                       'targets': ['support(P)']}]},
                      {'id': 'pauli_rotation_controlled_on_ancilla_one',
                       'operations': [{'angle': None,
                                       'controls': [],
                                       'gate': 'H for X; Sdg then H for Y; I for Z',
                                       'operation': 'basis_change_for_each_nonidentity_factor',
                                       'targets': ['support(P)']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'CNOT_LADDER',
                                       'operation': 'parity_compute',
                                       'targets': ['ordered support(P)']},
                                      {'angle': '2*theta for controlled exp(-i*theta*P); '
                                                'decompose CRZ into two CNOTs and one-qubit RZ '
                                                'rotations',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CRZ',
                                       'operation': 'controlled_central_rotation',
                                       'targets': ['last qubit in ordered support(P)']},
                                      {'angle': None,
                                       'controls': [],
                                       'gate': 'INVERSE_OF_PREVIOUS_CLIFFORDS',
                                       'operation': 'parity_uncompute_and_inverse_basis_change',
                                       'targets': ['support(P)']}]},
                      {'id': 'controlled_endpoint_pauli_string',
                       'operations': [{'angle': 'pi modulo the standard controlled-P phase '
                                                'convention',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CONTROLLED_PAULI_STRING',
                                       'operation': 'controlled_pauli_string_via_parity',
                                       'targets': ['support(P)']}]},
                      {'id': 'scalar_potential_coefficients',
                       'operations': [{'angle': 'a_Ns=-(lattice_spacing*bare_gauge_coupling/2)*sum_ell[(indicator(N<=ell)-1/2)*(indicator(s<=ell)-1/2)]',
                                       'controls': [],
                                       'gate': 'CLASSICAL_FORMULA',
                                       'operation': 'classical_coefficient_loop',
                                       'targets': ['N=0,...,4', 's=0,...,4', 'ell=0,...,3']}]},
                      {'id': 'controlled_D_N_sigma',
                       'operations': [{'angle': 'sigma*a_Ns, with sigma=+1 for U and sigma=-1 '
                                                'for U^-1',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CRZ',
                                       'operation': 'for_each_site_component',
                                       'targets': ['q[2*s]', 'q[2*s+1] for s=0,...,4']}]},
                      {'id': 'hamiltonian_offdiagonal_charge_preserving_blocks',
                       'operations': [{'angle': 'both coefficients are 3/4',
                                       'controls': [],
                                       'gate': 'COMMUTING_PAULI_PAIR',
                                       'operation': 'onsite_block_loop',
                                       'targets': ['for s=0,...,4: (X[2s]X[2s+1], '
                                                   'Y[2s]Y[2s+1])']},
                                      {'angle': 'coefficients (+1/4,-1/4), respectively',
                                       'controls': [],
                                       'gate': 'COMMUTING_PAULI_PAIR',
                                       'operation': 'link_block_loop_pair_1',
                                       'targets': ['for ell=0,...,3 with q=2*ell: '
                                                   '(X[q]Z[q+1]Y[q+2], Y[q]Z[q+1]X[q+2])']},
                                      {'angle': 'coefficients (-1/4,+1/4), respectively',
                                       'controls': [],
                                       'gate': 'COMMUTING_PAULI_PAIR',
                                       'operation': 'link_block_loop_pair_2',
                                       'targets': ['for ell=0,...,3 with q=2*ell: '
                                                   '(X[q+1]Z[q+2]Y[q+3], Y[q+1]Z[q+2]X[q+3])']},
                                      {'angle': 'both coefficients are -1/4',
                                       'controls': [],
                                       'gate': 'COMMUTING_PAULI_PAIR',
                                       'operation': 'link_block_loop_pair_3',
                                       'targets': ['for ell=0,...,3 with q=2*ell: '
                                                   '(X[q]Z[q+1]Z[q+2]X[q+3], '
                                                   'Y[q]Z[q+1]Z[q+2]Y[q+3])']},
                                      {'angle': 'both coefficients are -1/4',
                                       'controls': [],
                                       'gate': 'COMMUTING_PAULI_PAIR',
                                       'operation': 'link_block_loop_pair_4',
                                       'targets': ['for ell=0,...,3 with q=2*ell: '
                                                   '(X[q+1]X[q+2], Y[q+1]Y[q+2])']}]},
                      {'id': 'hamiltonian_diagonal_block',
                       'operations': [{'angle': 'for evolution interval tau, each term h_P*P '
                                                'uses theta=h_P*tau',
                                       'controls': [],
                                       'gate': 'PAULI_TERM_LIST',
                                       'operation': 'collect_authoritative_diagonal_terms',
                                       'targets': ['every nonidentity Agent-3 Hamiltonian term '
                                                   'whose Pauli label contains only I and Z, '
                                                   'preserving its declared coefficient and '
                                                   'qiskit_little_endian indexing']},
                                      {'angle': 'omitted because its phases cancel exactly '
                                                'between E(t)^dagger and E(t); no Hamiltonian '
                                                'shift is introduced',
                                       'controls': [],
                                       'gate': 'NO_GATE',
                                       'operation': 'exclude_identity_phase_only_inside_heisenberg_conjugations',
                                       'targets': ['Agent-3 identity term (5/16) I']}]},
                      {'id': 'controlled_E_t',
                       'operations': [{'angle': 'repeat N_T=trotter_steps times with '
                                                'delta=t/N_T; all Pauli rotations use '
                                                'pauli_rotation_controlled_on_ancilla_one',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CONTROLLED_S2',
                                       'operation': 'repeat_symmetric_step',
                                       'targets': ['q[0] through q[9]']}]},
                      {'id': 'controlled_E_t_dagger',
                       'operations': [{'angle': 'controlled_E_t with all gates reversed and '
                                                'angles negated',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CONTROLLED_S2_DAGGER',
                                       'operation': 'inverse_repeat_symmetric_step',
                                       'targets': ['q[0] through q[9]']}]},
                      {'id': 'controlled_wilson_dispatch',
                       'operations': [{'angle': 'ordered left-to-right exactly as Agent 3; '
                                                'circuit application is right-to-left',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CONTROLLED_WILSON_PRODUCT',
                                       'operation': 'sample_conditioned_ordered_compose',
                                       'targets': ['z=-2: '
                                                   '[E(-2)^dagger,D_0(-1),E(-2),E(-1)^dagger,D_1(-1),E(-1)]',
                                                   'z=-1: [E(-1)^dagger,D_1(-1),E(-1)]',
                                                   'z=0: []',
                                                   'z=1: [D_2(+1)]',
                                                   'z=2: '
                                                   '[E(1)^dagger,D_3(+1),E(1),D_2(+1)]']}]},
                      {'id': 'controlled_trajectory_V_ab',
                       'operations': [{'angle': 'z=sample_values.z1; j(z)=2*(z+2)+1; a,b '
                                                'selected by trajectory label',
                                       'controls': ['ancilla=1'],
                                       'gate': 'CONTROLLED_UNITARY_TRAJECTORY',
                                       'operation': 'compose_in_operator_left_to_right_order',
                                       'targets': ['E(z)^dagger',
                                                   'L_{j(z),a}',
                                                   'E(z)',
                                                   'W_z',
                                                   'R_b']}]}],
 'reconstruction': {'output_components': ['real', 'imag'],
                    'rule': 'For every valid selected sample, let r_ab be the expectation from '
                            'sample_z1__ab_real and s_ab the expectation from '
                            'sample_z1__ab_imag. The branch convention is '
                            '(|0>|phi>+|1>V_ab|psi>)/sqrt(2), so r_ab=Re(<phi|V_ab|psi>) and '
                            'Sdg then H gives s_ab=+Im(<phi|V_ab|psi>). With lambda_XX=1/2, '
                            'lambda_XY=i/2, lambda_YX=-i/2, lambda_YY=1/2, reconstruct '
                            'Re(M)=1/2*(r_XX-s_XY+s_YX+r_YY) and '
                            'Im(M)=1/2*(s_XX+r_XY-r_YX+s_YY). These coefficients already '
                            'include the observable prefactor 2 and both Jordan-Wigner factors '
                            '1/2.',
                    'terms': [{'coefficient': 'Re output: +1/2; Im output: 0',
                               'recipe_id': 'sample_z1__XX_real'},
                              {'coefficient': 'Re output: 0; Im output: +1/2',
                               'recipe_id': 'sample_z1__XX_imag'},
                              {'coefficient': 'Re output: 0; Im output: +1/2',
                               'recipe_id': 'sample_z1__XY_real'},
                              {'coefficient': 'Re output: -1/2; Im output: 0',
                               'recipe_id': 'sample_z1__XY_imag'},
                              {'coefficient': 'Re output: 0; Im output: -1/2',
                               'recipe_id': 'sample_z1__YX_real'},
                              {'coefficient': 'Re output: +1/2; Im output: 0',
                               'recipe_id': 'sample_z1__YX_imag'},
                              {'coefficient': 'Re output: +1/2; Im output: 0',
                               'recipe_id': 'sample_z1__YY_real'},
                              {'coefficient': 'Re output: 0; Im output: +1/2',
                               'recipe_id': 'sample_z1__YY_imag'}]},
 'schema_version': '1.1',
 'state_labels': ['phi', 'psi'],
 'state_preparation_interface': {'required_operations': ['prepare_phi',
                                                         'controlled_inverse_phi_on_ancilla_one',
                                                         'controlled_prepare_psi_on_ancilla_one'],
                                 'treat_as_black_box': True},
 'time_evolution': {'method': 'charge-preserving grouped symmetric second-order Suzuki product '
                              'formula',
                    'recommended_test_steps': [1, 2, 4, 8, 16],
                    'required': True,
                    'step_operations': [{'angle': 'for each Pauli term h_P*P in B_k use '
                                                  'theta=h_P*t/(2*N_T)',
                                         'gate': 'PAULI_ROTATION_SEQUENCE',
                                         'operation': 'ordered_forward_half_steps',
                                         'targets': ['B_1 through B_21 in the exact order: '
                                                     'five onsite blocks for s=0,...,4, '
                                                     'followed for ell=0,...,3 by link pairs '
                                                     '1,2,3,4']},
                                        {'angle': 'for each diagonal term h_P*P use '
                                                  'theta=h_P*t/N_T',
                                         'gate': 'PAULI_ROTATION_SEQUENCE',
                                         'operation': 'diagonal_full_step',
                                         'targets': ['all 33 nonidentity I/Z-only Agent-3 '
                                                     'Hamiltonian terms in their authoritative '
                                                     'listed order']},
                                        {'angle': 'for each Pauli term h_P*P in B_k use '
                                                  'theta=h_P*t/(2*N_T)',
                                         'gate': 'PAULI_ROTATION_SEQUENCE',
                                         'operation': 'ordered_reverse_half_steps',
                                         'targets': ['B_21 through B_1']},
                                        {'angle': 'repeat exactly N_T times; negative t '
                                                  'supplies negative rotation angles',
                                         'gate': 'LOOP',
                                         'operation': 'repeat_complete_step',
                                         'targets': ['the preceding forward, diagonal, and '
                                                     'reverse operations']}],
                    'step_symbol': 'N_T',
                    'symmetry_preservation_rule': 'At every finite N_T, each off-diagonal '
                                                  'block is a number-conserving commuting '
                                                  'Pauli pair and every diagonal block '
                                                  'commutes with Q_lat, so Q_lat and the '
                                                  'mandatory Q=0 to Q=-1 to Q=-1 to Q=0 '
                                                  'ordered sector path are exact. The '
                                                  'reconstructed Gauss generators remain zero '
                                                  'operator identities. A fixed product '
                                                  'ordering may break C, P, or T separately at '
                                                  'finite N_T; measure their corresponding '
                                                  'convergence diagnostics if used, and '
                                                  'require convergence as N_T increases. '
                                                  'Broken spatial translations and '
                                                  'continuum-only boosts are not imposed.'}}

OBSERVABLE_SPECIFICATION = {'coordinate_convention': {'mode': 'lattice_indices',
                           'spacing_parameter': 'lattice_spacing_a',
                           'temporal_spacing': 'same_as_spatial',
                           'units': 'natural_units_c_1'},
 'expression': '\\bar{\\psi}(zn) \\gamma^- W(zn,0) \\psi(0), n = (1,1), \\gamma^- = \\gamma^0 '
               '- \\gamma^1',
 'gauge_transporters': [{'from': '\\psi',
                         'id': 'Wlison line',
                         'path': 'straight lattice path',
                         'to': '\\bar{\\psi}'}],
 'insertions': [{'id': 'psibar', 'operator': '\\bar{\\psi}', 'position': {'t1': 'z1'}},
                {'id': 'psi', 'operator': '\\psi', 'position': {'t2': 'z2'}}],
 'name': 'user_observable',
 'observable_schema_version': '1.0',
 'ordering': 'as_written',
 'requested_components': ['real', 'imag'],
 'sampling': {'maximum_points': 10000,
              'mode': 'cartesian_product',
              'variables': {'z1': [-2, -1, 0, 1, 2], 'z2': [0]}}}

_NUM_SYSTEM_QUBITS = 10
_STATE_LABELS = ("phi", "psi")
_SUPPORTED_Z = (-2, -1, 0, 1, 2)
_TRAJECTORIES = ("XX", "XY", "YX", "YY")
_COMPONENTS = ("real", "imag")
_RECIPE_IDS = tuple(
    f"sample_z1__{trajectory}_{component}"
    for trajectory in _TRAJECTORIES
    for component in _COMPONENTS
)

_BASE_DIAGONAL_TERMS = (
    (1 / 16, "IIIIIIIIZZ"),
    (1 / 16, "IIIIIIZZII"),
    (1 / 16, "IIIIZZIIII"),
    (1 / 16, "IIZZIIIIII"),
    (1 / 16, "ZZIIIIIIII"),
    (1 / 32, "IIIIIIIZIZ"),
    (1 / 32, "IIIIIIZIIZ"),
    (1 / 32, "IIIIIIIZZI"),
    (1 / 32, "IIIIIIZIZI"),
    (1 / 32, "IIIIIZIZII"),
    (1 / 32, "IIIIZIIZII"),
    (1 / 32, "IIIIIZZIII"),
    (1 / 32, "IIIIZIZIII"),
    (1 / 32, "IIIZIZIIII"),
    (1 / 32, "IIZIIZIIII"),
    (1 / 32, "IIIZZIIIII"),
    (1 / 32, "IIZIZIIIII"),
    (1 / 32, "IZIZIIIIII"),
    (1 / 32, "ZIIZIIIIII"),
    (1 / 32, "IZZIIIIIII"),
    (1 / 32, "ZIZIIIIIII"),
    (-1 / 32, "IIIZIIIIIZ"),
    (-1 / 32, "IIZIIIIIIZ"),
    (-1 / 32, "IIIZIIIIZI"),
    (-1 / 32, "IIZIIIIIZI"),
    (-1 / 32, "IZIIIIIZII"),
    (-1 / 32, "ZIIIIIIZII"),
    (-1 / 32, "IZIIIIZIII"),
    (-1 / 32, "ZIIIIIZIII"),
    (-1 / 16, "IZIIIIIIIZ"),
    (-1 / 16, "ZIIIIIIIIZ"),
    (-1 / 16, "IZIIIIIIZI"),
    (-1 / 16, "ZIIIIIIIZI"),
)

_BARE_GAUGE_COUPLING = None
_BARE_FERMION_MASS = None
_LATTICE_SPACING = None
_TEMPORAL_LATTICE_SPACING = None
_WILSON_PARAMETER = None
_LIGHTCONE_NORMALIZATION = None
_IDENTITY_COEFFICIENT = None
_OFFDIAGONAL_BLOCKS = ()
_DIAGONAL_TERMS = ()
_SCALAR_POTENTIAL_COEFFICIENTS = ()
_RECONSTRUCTION_WEIGHTS = {}


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _selected_parameters(parameter_contract):
    if not isinstance(parameter_contract, dict):
        raise TypeError("parameter_contract must be a dict")
    required_top = {
        "constraints",
        "discretizations",
        "external_states",
        "model",
        "parameters",
        "schema_version",
        "validation_obligations",
    }
    if set(parameter_contract) != required_top:
        raise ValueError("parameter_contract top-level structure does not match")
    parameters = parameter_contract["parameters"]
    if not isinstance(parameters, list):
        raise TypeError("parameter_contract['parameters'] must be a list")
    result = {}
    for entry in parameters:
        if not isinstance(entry, dict) or "name" not in entry or "selected_value" not in entry:
            raise ValueError("every parameter entry must contain name and selected_value")
        name = entry["name"]
        if name in result:
            raise ValueError(f"duplicate parameter {name!r}")
        result[name] = entry["selected_value"]
    return result


def _validate_sample(sample_values):
    if not isinstance(sample_values, dict):
        raise TypeError("sample_values must be a dict")
    if set(sample_values) != {"z1", "z2"}:
        raise ValueError("sample_values must contain exactly z1 and z2")
    z1 = sample_values["z1"]
    z2 = sample_values["z2"]
    if isinstance(z1, bool) or not isinstance(z1, int) or z1 not in _SUPPORTED_Z:
        raise ValueError("sample_values['z1'] must be one of -2,-1,0,1,2")
    if isinstance(z2, bool) or not isinstance(z2, int) or z2 != 0:
        raise ValueError("sample_values['z2'] must equal 0")
    return z1


def _pauli_label(assignments):
    chars = ["I"] * _NUM_SYSTEM_QUBITS
    for qubit, pauli in assignments.items():
        if not isinstance(qubit, int) or not 0 <= qubit < _NUM_SYSTEM_QUBITS:
            raise ValueError("invalid Pauli qubit index")
        if pauli not in ("X", "Y", "Z"):
            raise ValueError("invalid Pauli factor")
        chars[_NUM_SYSTEM_QUBITS - 1 - qubit] = pauli
    return "".join(chars)


def _build_offdiagonal_blocks(mass, spacing, wilson):
    onsite = (mass + wilson / spacing) / 2.0
    derivative = 1.0 / (4.0 * spacing)
    wilson_link = wilson / (4.0 * spacing)
    blocks = []
    for site in range(5):
        q = 2 * site
        blocks.append((
            (onsite, _pauli_label({q: "X", q + 1: "X"})),
            (onsite, _pauli_label({q: "Y", q + 1: "Y"})),
        ))
    for ell in range(4):
        q = 2 * ell
        blocks.extend((
            (
                (derivative, _pauli_label({q: "X", q + 1: "Z", q + 2: "Y"})),
                (-derivative, _pauli_label({q: "Y", q + 1: "Z", q + 2: "X"})),
            ),
            (
                (-derivative, _pauli_label({q + 1: "X", q + 2: "Z", q + 3: "Y"})),
                (derivative, _pauli_label({q + 1: "Y", q + 2: "Z", q + 3: "X"})),
            ),
            (
                (-wilson_link, _pauli_label({
                    q: "X", q + 1: "Z", q + 2: "Z", q + 3: "X"
                })),
                (-wilson_link, _pauli_label({
                    q: "Y", q + 1: "Z", q + 2: "Z", q + 3: "Y"
                })),
            ),
            (
                (-wilson_link, _pauli_label({q + 1: "X", q + 2: "X"})),
                (-wilson_link, _pauli_label({q + 1: "Y", q + 2: "Y"})),
            ),
        ))
    if len(blocks) != 21:
        raise RuntimeError("failed to construct the required 21 Hamiltonian blocks")
    return tuple(blocks)


def _build_scalar_coefficients(spacing, coupling):
    rows = []
    for n in range(5):
        row = []
        for site in range(5):
            total = 0.0
            for ell in range(4):
                left = (1.0 if n <= ell else 0.0) - 0.5
                right = (1.0 if site <= ell else 0.0) - 0.5
                total += left * right
            row.append(-(spacing * coupling / 2.0) * total)
        rows.append(tuple(row))
    return tuple(rows)


def configure_runtime(parameter_contract, observable_specification):
    global _BARE_GAUGE_COUPLING
    global _BARE_FERMION_MASS
    global _LATTICE_SPACING
    global _TEMPORAL_LATTICE_SPACING
    global _WILSON_PARAMETER
    global _LIGHTCONE_NORMALIZATION
    global _IDENTITY_COEFFICIENT
    global _OFFDIAGONAL_BLOCKS
    global _DIAGONAL_TERMS
    global _SCALAR_POTENTIAL_COEFFICIENTS
    global _RECONSTRUCTION_WEIGHTS

    parameters = _selected_parameters(parameter_contract)

    expected_names = {
        "bare_gauge_coupling",
        "bare_fermion_mass",
        "lattice_spacing",
        "temporal_lattice_spacing",
        "wilson_parameter",
        "charge_reference_occupancy",
        "lightcone_normalization",
        "bare coupling: g",
        "bare quark mass: m",
        "lattice spacing: a",
        "number of lattice site: L",
    }
    if set(parameters) != expected_names:
        raise ValueError("parameter names do not match the generated program")

    model = parameter_contract["model"]
    if model != {
        "gauge_structure": "local U(1)",
        "name": "Schwinger model",
        "spacetime_dimension": "1+1",
    }:
        raise ValueError("model structure does not match the Schwinger adapter")

    states = parameter_contract["external_states"]
    if states.get("labels") != list(_STATE_LABELS):
        raise ValueError("external state labels must be exactly ['phi','psi']")
    if set(states.get("definitions", {})) != set(_STATE_LABELS):
        raise ValueError("external state definitions must be exactly phi and psi")
    if states.get("equivalent_groups") != []:
        raise ValueError("phi and psi must remain distinct external states")

    discretizations = parameter_contract["discretizations"]
    if not isinstance(discretizations, list) or len(discretizations) != 3:
        raise ValueError("exactly three discretization sectors are required")
    sectors = {entry.get("sector"): entry for entry in discretizations}
    if set(sectors) != {"spacetime", "fermion", "gauge"}:
        raise ValueError("discretization sectors do not match")
    spacetime = sectors["spacetime"]["settings"]
    fermion = sectors["fermion"]["settings"]
    gauge = sectors["gauge"]["settings"]
    if (
        spacetime.get("spatial_site_count") != 5
        or spacetime.get("spatial_link_count") != 4
        or spacetime.get("spatial_boundary_condition") != "open"
        or spacetime.get("observable_site_map") != "s(z)=z+2"
        or fermion.get("component_count_per_site") != 2
        or fermion.get("fermionic_mode_count") != 10
        or gauge.get("effective_qubit_count") != 10
        or gauge.get("independent_gauge_degree_count") != 0
    ):
        raise ValueError("lattice/qubit layout does not match the generated program")

    if not isinstance(observable_specification, dict):
        raise TypeError("observable_specification must be a dict")
    expected_spec_top = {
        "coordinate_convention",
        "expression",
        "gauge_transporters",
        "insertions",
        "name",
        "observable_schema_version",
        "ordering",
        "requested_components",
        "sampling",
    }
    if set(observable_specification) != expected_spec_top:
        raise ValueError("observable specification top-level structure does not match")
    if observable_specification["name"] != "user_observable":
        raise ValueError("unsupported observable name")
    if observable_specification["ordering"] != "as_written":
        raise ValueError("observable ordering must be as_written")
    if observable_specification["requested_components"] != ["real", "imag"]:
        raise ValueError("requested components must be exactly real and imag")
    if observable_specification["coordinate_convention"] != {
        "mode": "lattice_indices",
        "spacing_parameter": "lattice_spacing_a",
        "temporal_spacing": "same_as_spatial",
        "units": "natural_units_c_1",
    }:
        raise ValueError("coordinate convention does not match")
    if observable_specification["insertions"] != [
        {"id": "psibar", "operator": "\\bar{\\psi}", "position": {"t1": "z1"}},
        {"id": "psi", "operator": "\\psi", "position": {"t2": "z2"}},
    ]:
        raise ValueError("observable insertion structure does not match")
    transporters = observable_specification["gauge_transporters"]
    if transporters != [{
        "from": "\\psi",
        "id": "Wlison line",
        "path": "straight lattice path",
        "to": "\\bar{\\psi}",
    }]:
        raise ValueError("gauge transporter structure does not match")
    sampling = observable_specification["sampling"]
    if (
        sampling.get("mode") != "cartesian_product"
        or sampling.get("variables") != {"z1": list(_SUPPORTED_Z), "z2": [0]}
    ):
        raise ValueError("sampling assignments do not match the generated program")

    coupling = _finite_real(parameters["bare_gauge_coupling"], "bare_gauge_coupling")
    mass = _finite_real(parameters["bare_fermion_mass"], "bare_fermion_mass")
    spacing = _finite_real(parameters["lattice_spacing"], "lattice_spacing")
    temporal = _finite_real(
        parameters["temporal_lattice_spacing"], "temporal_lattice_spacing"
    )
    wilson = _finite_real(parameters["wilson_parameter"], "wilson_parameter")
    lightcone = _finite_real(
        parameters["lightcone_normalization"], "lightcone_normalization"
    )

    if coupling < 0.0:
        raise ValueError("bare_gauge_coupling must be nonnegative")
    if spacing <= 0.0 or temporal <= 0.0:
        raise ValueError("lattice spacings must be positive")
    if temporal != spacing:
        raise ValueError("temporal_lattice_spacing must equal lattice_spacing")
    if wilson == 0.0:
        raise ValueError("wilson_parameter must be nonzero")
    if parameters["charge_reference_occupancy"] != 1:
        raise ValueError("charge_reference_occupancy must equal 1")
    if lightcone != 1.0:
        raise ValueError("lightcone_normalization must equal 1")
    if parameters["number of lattice site: L"] != 5:
        raise ValueError("number of lattice site: L must equal 5")

    immutable_coupling = _finite_real(parameters["bare coupling: g"], "bare coupling: g")
    immutable_mass = _finite_real(parameters["bare quark mass: m"], "bare quark mass: m")
    immutable_spacing = _finite_real(parameters["lattice spacing: a"], "lattice spacing: a")
    if immutable_coupling != coupling:
        raise ValueError("bare coupling parameter assignments disagree")
    if immutable_mass != mass:
        raise ValueError("bare mass parameter assignments disagree")
    if immutable_spacing != spacing:
        raise ValueError("lattice spacing parameter assignments disagree")

    _BARE_GAUGE_COUPLING = coupling
    _BARE_FERMION_MASS = mass
    _LATTICE_SPACING = spacing
    _TEMPORAL_LATTICE_SPACING = temporal
    _WILSON_PARAMETER = wilson
    _LIGHTCONE_NORMALIZATION = lightcone

    coulomb_scale = 4.0 * spacing * coupling * coupling
    _IDENTITY_COEFFICIENT = (5.0 / 16.0) * coulomb_scale
    _DIAGONAL_TERMS = tuple(
        (base_coefficient * coulomb_scale, label)
        for base_coefficient, label in _BASE_DIAGONAL_TERMS
    )
    _OFFDIAGONAL_BLOCKS = _build_offdiagonal_blocks(mass, spacing, wilson)
    _SCALAR_POTENTIAL_COEFFICIENTS = _build_scalar_coefficients(spacing, coupling)
    _RECONSTRUCTION_WEIGHTS = {
        "sample_z1__XX_real": 0.5 + 0.0j,
        "sample_z1__XX_imag": 0.0 + 0.5j,
        "sample_z1__XY_real": 0.0 + 0.5j,
        "sample_z1__XY_imag": -0.5 + 0.0j,
        "sample_z1__YX_real": 0.0 - 0.5j,
        "sample_z1__YX_imag": 0.5 + 0.0j,
        "sample_z1__YY_real": 0.5 + 0.0j,
        "sample_z1__YY_imag": 0.0 + 0.5j,
    }


def _append_controlled_pauli_rotation(circuit, ancilla, label, theta):
    if len(label) != _NUM_SYSTEM_QUBITS:
        raise ValueError("Hamiltonian Pauli label has the wrong length")
    support = []
    factors = {}
    for qubit in range(_NUM_SYSTEM_QUBITS):
        factor = label[_NUM_SYSTEM_QUBITS - 1 - qubit]
        if factor != "I":
            support.append(qubit)
            factors[qubit] = factor
    if not support:
        return

    for qubit in support:
        factor = factors[qubit]
        if factor == "X":
            circuit.h(qubit)
        elif factor == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        elif factor != "Z":
            raise ValueError("unsupported Pauli factor")

    for left, right in zip(support[:-1], support[1:]):
        circuit.cx(left, right)

    circuit.crz(2.0 * theta, ancilla, support[-1])

    for left, right in reversed(tuple(zip(support[:-1], support[1:]))):
        circuit.cx(left, right)

    for qubit in reversed(support):
        factor = factors[qubit]
        if factor == "X":
            circuit.h(qubit)
        elif factor == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def _append_controlled_evolution(circuit, ancilla, time, trotter_steps):
    delta = float(time) / trotter_steps
    for _ in range(trotter_steps):
        for block in _OFFDIAGONAL_BLOCKS:
            for coefficient, label in block:
                _append_controlled_pauli_rotation(
                    circuit, ancilla, label, coefficient * delta / 2.0
                )
        for coefficient, label in _DIAGONAL_TERMS:
            _append_controlled_pauli_rotation(
                circuit, ancilla, label, coefficient * delta
            )
        for block in reversed(_OFFDIAGONAL_BLOCKS):
            for coefficient, label in block:
                _append_controlled_pauli_rotation(
                    circuit, ancilla, label, coefficient * delta / 2.0
                )


def _append_controlled_evolution_dagger(circuit, ancilla, time, trotter_steps):
    temporary = QuantumCircuit(_NUM_SYSTEM_QUBITS + 1)
    _append_controlled_evolution(temporary, ancilla, time, trotter_steps)
    circuit.compose(temporary.inverse(), inplace=True)


def _append_controlled_endpoint_pauli(circuit, ancilla, label):
    controlled = PauliGate(label).control(1)
    circuit.append(controlled, [ancilla] + list(range(_NUM_SYSTEM_QUBITS)))


def _append_controlled_D(circuit, ancilla, site_n, sigma):
    if site_n not in range(5) or sigma not in (-1, 1):
        raise ValueError("invalid scalar-potential dispatch")
    for site in range(5):
        angle = sigma * _SCALAR_POTENTIAL_COEFFICIENTS[site_n][site]
        circuit.crz(angle, ancilla, 2 * site)
        circuit.crz(angle, ancilla, 2 * site + 1)


def _append_wilson(circuit, ancilla, z, trotter_steps):
    if z == -2:
        operations = (
            ("E_DAGGER", -2),
            ("D", 0, -1),
            ("E", -2),
            ("E_DAGGER", -1),
            ("D", 1, -1),
            ("E", -1),
        )
    elif z == -1:
        operations = (
            ("E_DAGGER", -1),
            ("D", 1, -1),
            ("E", -1),
        )
    elif z == 0:
        operations = ()
    elif z == 1:
        operations = (("D", 2, 1),)
    elif z == 2:
        operations = (
            ("E_DAGGER", 1),
            ("D", 3, 1),
            ("E", 1),
            ("D", 2, 1),
        )
    else:
        raise ValueError("unsupported Wilson trajectory")

    for operation in reversed(operations):
        if operation[0] == "E":
            _append_controlled_evolution(
                circuit, ancilla, operation[1], trotter_steps
            )
        elif operation[0] == "E_DAGGER":
            _append_controlled_evolution_dagger(
                circuit, ancilla, operation[1], trotter_steps
            )
        elif operation[0] == "D":
            _append_controlled_D(
                circuit, ancilla, operation[1], operation[2]
            )
        else:
            raise RuntimeError("untranslatable Wilson operation")


def _append_trajectory(circuit, ancilla, z, left_factor, right_factor, trotter_steps):
    right_label = _pauli_label({
        0: "Z",
        1: "Z",
        2: "Z",
        3: "Z",
        4: "Z",
        5: right_factor,
    })
    _append_controlled_endpoint_pauli(circuit, ancilla, right_label)
    _append_wilson(circuit, ancilla, z, trotter_steps)
    _append_controlled_evolution(circuit, ancilla, z, trotter_steps)

    endpoint = 2 * (z + 2) + 1
    left_assignments = {qubit: "Z" for qubit in range(endpoint)}
    left_assignments[endpoint] = left_factor
    left_label = _pauli_label(left_assignments)
    _append_controlled_endpoint_pauli(circuit, ancilla, left_label)

    _append_controlled_evolution_dagger(circuit, ancilla, z, trotter_steps)


def build_observable_measurements(state_circuits, trotter_steps, sample_values):
    z = _validate_sample(sample_values)

    if isinstance(trotter_steps, bool) or not isinstance(trotter_steps, int):
        raise TypeError("trotter_steps must be a positive integer")
    if trotter_steps <= 0:
        raise ValueError("trotter_steps must be a positive integer")

    if not isinstance(state_circuits, dict):
        raise TypeError("state_circuits must be a dict")
    if set(state_circuits) != set(_STATE_LABELS):
        raise ValueError("state_circuits must map exactly phi and psi")

    for label in _STATE_LABELS:
        state = state_circuits[label]
        if not isinstance(state, QuantumCircuit):
            raise TypeError(f"state_circuits[{label!r}] must be a QuantumCircuit")
        if state.num_qubits != _NUM_SYSTEM_QUBITS:
            raise ValueError(f"state_circuits[{label!r}] must have 10 qubits")
        if state.num_clbits != 0:
            raise ValueError(f"state_circuits[{label!r}] must have no classical bits")
        if state.parameters:
            raise ValueError(f"state_circuits[{label!r}] must be parameter-free")

    phi = state_circuits["phi"].copy()
    psi = state_circuits["psi"].copy()
    phi_inverse_controlled = phi.to_gate(label="prepare_phi").inverse().control(
        1, label="controlled_inverse_phi"
    )
    psi_controlled = psi.to_gate(label="prepare_psi").control(
        1, label="controlled_prepare_psi"
    )

    recipes = []
    ancilla = _NUM_SYSTEM_QUBITS
    all_qubits = list(range(_NUM_SYSTEM_QUBITS + 1))
    controlled_qargs = [ancilla] + list(range(_NUM_SYSTEM_QUBITS))

    for trajectory in _TRAJECTORIES:
        left_factor, right_factor = trajectory
        for component in _COMPONENTS:
            circuit = QuantumCircuit(_NUM_SYSTEM_QUBITS + 1)
            circuit.h(ancilla)
            circuit.compose(phi, qubits=range(_NUM_SYSTEM_QUBITS), inplace=True)
            circuit.append(phi_inverse_controlled, controlled_qargs)
            circuit.append(psi_controlled, controlled_qargs)

            _append_trajectory(
                circuit,
                ancilla,
                z,
                left_factor,
                right_factor,
                trotter_steps,
            )

            if component == "real":
                circuit.h(ancilla)
            else:
                circuit.sdg(ancilla)
                circuit.h(ancilla)

            if circuit.parameters:
                raise RuntimeError("constructed measurement circuit is not parameter-free")
            if circuit.num_clbits != 0:
                raise RuntimeError("constructed measurement circuit has classical bits")

            readout = ["I"] * len(all_qubits)
            readout[len(all_qubits) - 1 - ancilla] = "Z"
            recipe_id = f"sample_z1__{trajectory}_{component}"
            recipes.append({
                "recipe_id": recipe_id,
                "component": component,
                "circuit": circuit,
                "readout_pauli": "".join(readout),
            })

    if tuple(recipe["recipe_id"] for recipe in recipes) != _RECIPE_IDS:
        raise RuntimeError("measurement recipe order does not match the contract")
    return recipes


def reconstruct_observable(expectations, sample_values):
    _validate_sample(sample_values)
    if not isinstance(expectations, dict):
        raise TypeError("expectations must be a dict")
    if set(expectations) != set(_RECIPE_IDS):
        missing = sorted(set(_RECIPE_IDS) - set(expectations))
        extra = sorted(set(expectations) - set(_RECIPE_IDS))
        raise ValueError(f"expectation recipe IDs do not match; missing={missing}, extra={extra}")

    result = 0.0j
    for recipe_id in _RECIPE_IDS:
        value = expectations[recipe_id]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"expectations[{recipe_id!r}] must be a real float")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"expectations[{recipe_id!r}] must be finite")
        result += _RECONSTRUCTION_WEIGHTS[recipe_id] * value
    return complex(result)


configure_runtime(PARAMETER_CONTRACT, OBSERVABLE_SPECIFICATION)
