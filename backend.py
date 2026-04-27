import os

import requests
import numpy as np
from dotenv import load_dotenv
from iqm.qiskit_iqm import IQMProvider
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import BackendSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

def get_real_iqm_backend(env_file="token.env"):
    if os.path.exists(env_file):
        load_dotenv(env_file)
    else:
        print("BRAK TOKENU!")
        raise ValueError("Brak Tokenu")

    server_url = os.getenv("SERVER")

    if not server_url:
        raise ValueError("Brak zmiennej SERVER w środowisku!")

    print(f"Łączenie z serwerem: {server_url}...")

    provider = IQMProvider(server_url)
    os.environ["IQM_CLIENT_REQUEST_TIMEOUT"] = "100"
    backend = provider.get_backend()
    backend.client._request_timeout = 100
    backend.options.update_options(timeout=100)
    print(f"Połączono pomyślnie z: {backend.name}")
    return backend
def get_json_from_url(url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Błąd pobierania danych: {response.status_code} - {response.text}")


def get_calibrated_noise_model(url, token, verbose = False):
    data = get_json_from_url(url, token)
    if verbose:
        print(f"Calibration date: {data['created_timestamp']}")


    observations = data['observations']

    noise_model = NoiseModel()

    def get_qb_idx(qb_name):
        return int(qb_name.replace('QB', '')) - 1

    readout_data = {}

    for obs in observations:
        field = obs['dut_field']
        val = obs['value']

        if "metrics.ssro.measure.constant" in field:
            parts = field.split('.')
            qb_idx = get_qb_idx(parts[4])
            metric = parts[5]

            if qb_idx not in readout_data: readout_data[qb_idx] = {}
            if 'error_0_to_1' in metric: readout_data[qb_idx]['p01'] = val
            if 'error_1_to_0' in metric: readout_data[qb_idx]['p10'] = val


        elif "metrics.rb.clifford.xy" in field:
            qb_idx = get_qb_idx(field.split('.')[4])
            error_rate = 1 - val
            error = depolarizing_error(error_rate, 1)
            noise_model.add_quantum_error(error, ["h", "rx", "ry", "rz"], [qb_idx])

        elif "metrics.irb.cz" in field:
            qbs = field.split('.')[4].split('__')
            idx1, idx2 = get_qb_idx(qbs[0]), get_qb_idx(qbs[1])
            error_rate = 1 - val
            error = depolarizing_error(error_rate, 2)
            noise_model.add_quantum_error(error, ["cx", "cz"], [idx1, idx2])

    for qb_idx, errs in readout_data.items():
        if 'p01' in errs and 'p10' in errs:
            p01 = errs['p01']
            p10 = errs['p10']
            matrix = [[1 - p01, p01], [p10, 1 - p10]]
            noise_model.add_readout_error(ReadoutError(matrix), [qb_idx])

    return noise_model

def get_backend(backend="Odra5", num_features=3, verbose = False):
    if os.path.exists("token.env"):
        load_dotenv("token.env")
    else:
        print("BRAK TOKENU!")
        raise ValueError("Brak Tokenu")
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2)
    # feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1, entanglement='linear') # lub to
    kernel = None
    sampler = None
    odra_backend = None

    if backend == "Odra5":
        odra_backend = get_real_iqm_backend()

        sampler = BackendSampler(backend=odra_backend)
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    elif backend == "FakeOdra":
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
        noise_model.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["h", "rx", "ry", "rz"])

        coupling_map = CouplingMap([[0, 2], [1, 2], [2, 3], [2, 4]])

        base_backend = GenericBackendV2(num_qubits=5)

        noisy_sim = AerSimulator.from_backend(base_backend)

        noisy_sim.set_options(
            noise_model=noise_model,
            coupling_map=coupling_map
        )

        sampler = BackendSampler(
            backend=noisy_sim,
            options={"shots": 1024}
        )

        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
        odra_backend = noisy_sim
    elif backend == "FakeOdraRealTime":
        token = os.getenv("IQM_TOKEN")

        static_url = "https://odra5.e-science.pl/api/latest/calibration-sets/01966208-f3ec-73b7-890d-100000000000/default/metrics"
        noise_model = get_calibrated_noise_model(
            static_url,
            token,
            verbose=verbose
        )
        print(noise_model)
        noisy_sim = AerSimulator(noise_model=noise_model)
        coupling_map = CouplingMap([[0, 2], [1, 2], [2, 3], [2, 4]])

        noisy_sim.set_options(
            noise_model=noise_model,
            coupling_map=coupling_map
        )

        sampler = BackendSampler(
            backend=noisy_sim,
            options={"shots": 1024}
        )

        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
        odra_backend = noisy_sim
    else:
        from qiskit.primitives import Sampler
        sampler = Sampler()
        kernel = FidelityQuantumKernel(feature_map=feature_map)

    return kernel, sampler, feature_map, odra_backend
