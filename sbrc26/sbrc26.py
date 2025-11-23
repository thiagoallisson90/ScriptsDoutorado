from sklearn.cluster import KMeans
import numpy as np
import os
import threading
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime

result_path = '/home/thiago/SBRC26/RSFTPA2'
ns3_path = '/home/thiago/ns-3-allinone/ns-3.44'
#scratch_path = f'{ns3_path}/scratch'
ns3_cmd = f'{ns3_path}/./ns3'
script = 'scratch/sbrc26.cc'

def gen_sm_coords(n_sms=1000, axis=7000, seed=42):
	np.random.seed(seed)
	sm_coords = np.random.uniform(0, axis, (n_sms, 2))
	np.random.seed(None)

	return sm_coords

def gen_gw(sm_coords, k, seed=None):
  """
   centers, labels, avg_distances, avg_dist
  """
  clf = KMeans(k, random_state=seed).fit(sm_coords)

  centers = clf.cluster_centers_
  labels = clf.labels_

  distances = [0 for _ in range(0, k)]
  n = [0 for _ in range(0, k)]  
  for i, label in enumerate(labels):
    distances[label] = distances[label] + np.linalg.norm(sm_coords[i] - centers[label])
    n[label] = n[label] + 1

  avg_distances = [dist / n[i] for i, dist in enumerate(distances)]
  avg_dist = np.mean(avg_distances)

  return centers, labels, avg_distances, avg_dist

def run_simulation(ns3_cmd, script, params01, params02, params3, j):
    timestamp_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_start}] [INFO] Iniciando execução: run {j}")

    start_time = time.time()
    run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {params3} --nRun={j}"'
    exit_code = os.system(run_cmd)
    end_time = time.time()

    duration = round(end_time - start_time, 2)
    timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp_end}] [INFO] Finalizado run {j} | Código: {exit_code} | Duração: {duration}s")

    return duration

def simulate(script, path, sm_coords, gw_coords, radius, sfa, ns3_cmd,
             adr_enabled=0, adr_type="ns3::AdrComponent", adr_name="adr"):
    init_iters = [1]
    end_iters = [11]

    # Lista para armazenar duração de cada simulação
    durations = []

    # Detectar número de núcleos
    max_procs = os.cpu_count()
    print(f"[INFO] Detectado {max_procs} núcleos. É possível executar até {max_procs} simulações em paralelo.")

    # Salvar arquivos CSV
    sm_file = make_file_name(path, f'{len(sm_coords)}sm_file')
    pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)
    gw_file = make_file_name(path, f'{len(gw_coords)}gw_file')
    pd.DataFrame(gw_coords).to_csv(gw_file, header=False, index=False)

    # Parâmetros fixos
    params01 = f'--nDevices={len(sm_coords)} --nGateways={len(gw_coords)} --path={path}'
    params02 = f'--smFile={sm_file} --gwFile={gw_file} --radius={radius} --sfa={sfa}'
    params03 = f'--adrEnabled={adr_enabled} --adrType={adr_type} --adrName={adr_name}'

    # Rodar comando inicial do ns3 (se necessário)
    os.system(ns3_cmd)

    # Executor com processos
    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = []
        for i in range(len(init_iters)):
            for j in range(init_iters[i], end_iters[i]):
                futures.append(executor.submit(run_simulation, ns3_cmd, script, 
                                               params01, params02, params03, j))

        # Capturar a duração de cada simulação
        for future in futures:
            durations.append(future.result())

    print(f"\n[INFO] Durações de todas as simulações: {durations}")
    print('#'*100)
    return durations

"""def simulate_sfa(script, nDevices, nGateways, sfa, path, sm_coords, gw_coords, radius, 
                 tx_mode="nack", adr_enabled=0, adr_type="ns3::AdrComponent", adr_name="adr"):
  init_iters = [1]
  end_iters = [11]
  threads = []

  sm_file = make_file_name(path, f'{len(sm_coords)}sm_file')
  pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)
  gw_file = make_file_name(path, f'{len(gw_coords)}gw_file')
  pd.DataFrame(gw_coords).to_csv(gw_file, header=False, index=False)

  params01 = f'--nDevices={nDevices} --nGateways={nGateways} --sfa={sfa} --path={path} --radius={radius}'
  params02 = f'--smFile={sm_file} --gwFile={gw_file} --txMode={tx_mode} --adrEnabled={adr_enabled}'
  params03 = f'--adrType={adr_type} --adrName={adr_name}'

  for i in range(len(init_iters)):
    os.system(ns3_cmd)
    for j in range(init_iters[i], end_iters[i]):
      run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {params03} --nRun={j}"'  # --gdb
      t = threading.Thread(target=os.system, args=[run_cmd])
      threads.append(t)
      t.start()

    for t in threads:
      t.join()

  os.system(ns3_cmd)
"""

def make_file_name(path, name, ext='csv'):
   return f'{path}/{name}.{ext}'

def check(file):
   names = [
        'sent', 'rec', 'pdr', 'imr_sent', 'imr_rec', 'imr_pdr', 'an_sent', 'an_rec',
        'an_pdr', 'delay', 'imr_delay', 'pcc_delay', 'rssi', 'snr', 'energy', 'tput',
        'ee1', 'ee2', 'ee3', 'ee4', 'rssi_pkts', 'snr_pkts', 'nRun'
   ]

   df = pd.read_csv(file, names=names)
   print(f"IMR Min. PDR = {df['imr_pdr'].min()} and AN Min. PDR = {df['an_pdr'].min()}")

   return df['imr_pdr'].min() >= 99.5 and df['an_pdr'].min() >= 99.5

def test_isfa():
  sfa = 'isfa'
  scenarios = [200, 400, 600, 800, 1000]
  radius = 7000

  # ISFA
  k = 1
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')
     # k = ks
     path = f'{result_path}/{sfa}/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     
     # simulate_sfa(script, scenario, k, sfa, path, sm_coords, data[0], radius)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)

     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
                 sfa, ns3_cmd, adr_enabled=0)
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

def test_rsfa():
  sfa = 'rsfa'
  scenarios = [800, 1000]
  radius = 7000

  # RSFA
  ks = [4, 4]
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')
     k = ks[i]
     path = f'{result_path}/{sfa}/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     # simulate_sfa(script, scenario, k, sfa, path, sm_coords, data[0], radius)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)
     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
                 sfa, ns3_cmd, adr_enabled=0)
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

def test_adr():
  sfa = 'none'
  scenarios = [200, 400, 600, 800, 1000]
  radius = 7000

  # ADR
  ks = [1, 1, 1, 1, 1]
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')
     k = ks[i]
     path = f'{result_path}/adr/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=1, adr_name='adr')
     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
                 sfa, ns3_cmd, adr_enabled=1, adr_name='adr')
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

def test_sftpa():
  sfa = 'sftpa'
  scenarios = [200, 400, 600, 800, 1000]
  radius = 7000

  # SFTPA
  k = 1
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')

     #k = ks[i]

     path = f'{result_path}/{sfa}/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)
     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

     # k = 1

def test_sftpa2():
  sfa = 'sftpa2'
  scenarios = [200, 400, 600, 800, 1000]
  radius = 7000

  # SFTPA
  k = 1
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')

     #k = ks[i]

     path = f'{result_path}/{sfa}/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)
     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=0)
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

     # k = 1

def test_caadr():
  sfa = 'none'
  scenarios = [200, 400, 600, 800, 1000]
  radius = 7000

  # CA-ADR
  ks = [1, 1, 1, 1, 1]
  for i, scenario in enumerate(scenarios):
     print(f'{scenario} SMs')
     k = ks[i]
     path = f'{result_path}/caadr/{scenario}'
     sm_coords = gen_sm_coords(scenario)
     
     data = gen_gw(sm_coords, k, 42)
     simulate(script, path, sm_coords, data[0], radius, 
              sfa, ns3_cmd, adr_enabled=1, 
              adr_type="ns3::CAADR", adr_name='caadr')
     filesfa = make_file_name(path, f'{k}gw_data')
     
     while(check(filesfa) == False):
        k = k + 1
        if(k == 29):
           k = 28
           break
        data = gen_gw(sm_coords, k, 42) 
        simulate(script, path, sm_coords, data[0], radius, 
                 sfa, ns3_cmd, adr_enabled=1, 
                 adr_type="ns3::CAADR", adr_name='caadr')
        filesfa = make_file_name(path, f'{k}gw_data')

     pd.DataFrame([k]).to_csv(make_file_name(path, 'k'), header=False, index=False)
     pd.DataFrame(data[0]).to_csv(make_file_name(path, 'coords'), header=False, index=False)
     pd.DataFrame(data[1]).to_csv(make_file_name(path, 'labels'), header=False, index=False)
     pd.DataFrame(data[2]).to_csv(make_file_name(path, 'avg_dists'), header=False, index=False)
     pd.DataFrame([data[3]]).to_csv(make_file_name(path, 'avg_dist'), header=False, index=False)

if __name__ == '__main__':
  print("Tests for SBRC'26")

  # 99.5%

  # I-SFA
  # test_isfa()

  # RSFA
  # test_rsfa()

  # SFTPA
  # test_sftpa()

  # SFTPA2
  test_sftpa2()

  # ADR
  # test_adr()

  # CA-ADR
  # test_caadr()
