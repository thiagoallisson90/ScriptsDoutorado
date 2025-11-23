#!/usr/bin/python3

import os
import pandas as pd
import threading
import scipy.stats as stats
from typing import Union, Optional
import matplotlib.pyplot as plt
import numpy as np

result_path = '/home/thiago/Doutorado/Jisa2025'
ns3_path = '/home/thiago/ns-3-allinone/ns-3.43'
scratch_path = f'{ns3_path}/scratch'
ns3_cmd = f'{ns3_path}/./ns3'

from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
#from skfuzzy.cluster import cmeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plt
import time

def gen_sm_coords(n_sms=1000, axis=7000, seed=42):
	np.random.seed(seed)
	sm_coords = np.random.uniform(0, axis, (n_sms, 2))
	np.random.seed(None)

	return sm_coords

'''
def simulate(script, path, sm_coords, dap_coords, 
						 init_iters=[1, 6, 11, 16, 21, 26, 31],
						 end_iters=[6, 11, 16, 21, 26, 31, 34],
             radius=7000, labels=[], loss="log", sf="up", is_ack=False):
	sm_file = f'{path}/{len(sm_coords)}_sm_coords.csv'
	dap_file = f'{path}/{len(dap_coords)}_dap_coords.csv'

	pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)
	pd.DataFrame(dap_coords).to_csv(dap_file, header=False, index=False)		
	
	extra_params = ''
	if len(labels) > 0:
		labels_file = f'{path}/{len(dap_coords)}_labels.csv'
		pd.DataFrame(labels).to_csv(labels_file, header=False, index=False)		
		extra_params = f'--labels={labels_file}'

	params01 = f'--radius={radius} --path={path} --nDevices={len(sm_coords)} --lossModel={loss} --isAck={is_ack}'
	params02 = f'--smFile={sm_file} --gwFile={dap_file} --nGateways={len(dap_coords)} --modeToSetSf={sf}'
	
	threads = []
	for i in range(len(init_iters)):
		os.system(ns3_cmd)
		for j in range(init_iters[i], end_iters[i]):
			run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {extra_params} --nRun={j}"' #--gdb
			t = threading.Thread(target=os.system, args=[run_cmd])
			threads.append(t)
			t.start()

		for t in threads:
			t.join()
   
		os.system(ns3_cmd)
'''

def simulate_sfa(script, nDevices, nGateways, sfa, path, sm_file, gw_file, radius, 
                 tx_mode="nack", adr_enabled=0, adr_type="ns3::AdrComponent", adr_name=""):
  init_iters = [1, 6]
  end_iters = [6, 11]
  threads = []

  times = []

  params01 = f'--nDevices={nDevices} --nGateways={nGateways} --sfa={sfa} --path={path} --radius={radius}'
  params02 = f'--smFile={sm_file} --gwFile={gw_file} --txMode={tx_mode} --adrEnabled={adr_enabled}'
  params03 = f'--adrType={adr_type} --adrName={adr_name}'

  for i in range(len(init_iters)):
    os.system(ns3_cmd)
    start = time.perf_counter()

    for j in range(init_iters[i], end_iters[i]):
      run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {params03} --nRun={j}"'  # --gdb
      t = threading.Thread(target=os.system, args=[run_cmd])
      threads.append(t)
      t.start()

    for t in threads:
      t.join()
    
    end = time.perf_counter()
    times.append(round(end - start, 2))

  os.system(ns3_cmd)

  return times

'''import os
import time
from multiprocessing import Process

def simulate_sfa(script, nDevices, nGateways, sfa, path, sm_file, gw_file, radius, 
                 tx_mode="nack", adr_enabled=0, adr_type="ns3::AdrComponent", adr_name=""):
    init_iters = [1, 6]
    end_iters = [6, 11]
    
    times = []

    params01 = f'--nDevices={nDevices} --nGateways={nGateways} --sfa={sfa} --path={path} --radius={radius}'
    params02 = f'--smFile={sm_file} --gwFile={gw_file} --txMode={tx_mode} --adrEnabled={adr_enabled}'
    params03 = f'--adrType={adr_type} --adrName={adr_name}'

    for i in range(len(init_iters)):
        os.system(ns3_cmd)  # Build inicial
        start = time.perf_counter()

        processes = []

        for j in range(init_iters[i], end_iters[i]):
            run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {params03} --nRun={j}"'
            p = Process(target=os.system, args=(run_cmd,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        end = time.perf_counter()
        times.append(round(end - start, 2))

    os.system(ns3_cmd)  # Build final opcional

    return times'''

def ic_analysis(data):
    erro_padrao = round(stats.sem(data), 3)
    mean = round(data.mean(), 3)

    intervalo_conf = stats.t.interval(0.95, len(data)-1, loc=mean, scale=erro_padrao)
    margem_erro = round((intervalo_conf[1] - intervalo_conf[0]) / 2, 3)

    return {
        'mean': mean, 
        'error': erro_padrao, 
        'ic': (round(intervalo_conf[0], 3), round(intervalo_conf[1], 3)), 
        'margin': margem_erro
    }

def plot_curves(x: Union[list, np.array], y: Union[list, np.array], 
                xlabel: str, ylabel: str, labels: Union[list, np.array],
                figname='metric.png', unit='', fig_size=(12, 10), ext='png',
                fontsize=25, loc='best'):
    _, ax = plt.subplots(figsize=fig_size, dpi=100)

    # Definindo marcadores, estilos de linha e cores
    markers = ['^', 'o', 's']  # Apenas os marcadores
    line_styles = ['--', '-.', ':']  # Estilos de linha para diferenciação
    colors = ['#0072B2', '#E69F00', '#CC79A7']  # Cores amigáveis para daltonismo
    
    for i, value in enumerate(y):
        ax.plot(x, value, marker=markers[i], linestyle=line_styles[i], 
                label=labels[i], color=colors[i], linewidth=4, markersize=12)

    ax.grid(axis='y', linestyle='--', color='gray', alpha=1.0)  # Grid no eixo y
    ax.set_xticks(x)  # Definindo os ticks do eixo X com base nos valores de `x`

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(f'{ylabel} {"(" + unit + ")" if unit else ""}', fontsize=fontsize)
    ax.legend(fontsize=30, loc=loc)

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()

    # Salvando o gráfico no arquivo especificado
    # plt.savefig(figname, dpi=100)
    plt.savefig(figname, format=ext, dpi=100, bbox_inches='tight')
    plt.close('all')

def plot_curves_error(x: Union[list, np.array], 
                      y: Union[list, np.array], 
                      xlabel: str, ylabel: str, labels: Union[list, np.array],
                      yerr: Optional[Union[list, np.array]] = None,
                      figname='metric.png', unit='', fig_size=(12, 10), ext='png',
                      fontsize=25, loc='best'):
    _, ax = plt.subplots(figsize=fig_size, dpi=100)

    # Definindo marcadores, estilos de linha e cores
    markers = ['^', 'o', 's']  
    line_styles = ['--', '-.', ':']  
    colors = ['#0072B2', '#E69F00', '#CC79A7']  

    for i, value in enumerate(y):
        err = yerr[i] if yerr is not None else None  # Obtendo erro, se existir
        ax.errorbar(x, value, yerr=err, fmt=markers[i] + line_styles[i], 
                    label=labels[i], color=colors[i], linewidth=4, markersize=12, 
                    capsize=5, capthick=2, elinewidth=2)  

    ax.grid(axis='y', linestyle='--', color='gray', alpha=1.0)  
    ax.set_xticks(x)  

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(f'{ylabel} {"(" + unit + ")" if unit else ""}', fontsize=fontsize)
    ax.legend(fontsize=30, loc=loc)

    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(figname, format=ext, dpi=100, bbox_inches='tight')
    plt.close('all')

def make_file_name(path: str, n_gws: int, name: str, sfa: str, tx_mode = "nack", 
                   adr_enabled = 0, adr_name = '', extension: str = "csv") -> str:
    file_name = f"{path}/{n_gws}gw_" \
                f"{adr_name + '_' if adr_enabled == 1 else ''}" \
                f"{sfa + '_' if sfa else ''}" \
                f"{tx_mode}_{name}.{extension}"
    return file_name

def km_sbrc_nack(script, sm_coords, sm_file, sfa, path, radius, tx_mode='nack', 
                 adr_enabled=0, adr_type='ns3::AdrComponent', adr_name='adr', k=1):
    names = [
        'sent', 'rec', 'pdr', 'imr_sent', 'imr_rec', 'imr_pdr', 'billing_sent', 'billing_rec',
        'billing_pdr', 'delay', 'imr_delay', 'pcc_delay', 'rssi', 'snr', 'energy', 'tput',
        'ee1', 'ee2', 'ee3', 'ee4', 'rssi_pkts', 'snr_pkts', 'nRun'
    ]

    clf = None
    while k <= 15:
        gw_file = f'{path}/{k}gw_file.csv'
        clf = KMeans(k, n_init='auto', random_state=42)
        clf.fit(sm_coords)
        
        pd.DataFrame(clf.cluster_centers_).to_csv(gw_file, header=False, index=False)
        
        times = \
        simulate_sfa(
            script, len(sm_coords), k, sfa, path, sm_file, gw_file, 
            radius, tx_mode, adr_enabled, adr_type, adr_name
        )
        time_sum = round(np.sum(times), 2)
        print(f'{k} DAPs', times, time_sum)
        time_data = [t for t in times]
        time_data.append(time_sum)
        pd.DataFrame(time_data).to_csv(f'{path}/{k}gws_times.csv', header=False, index=False)
        
        file_name = make_file_name(path, k, 'data', sfa, tx_mode, adr_enabled, adr_name)
        df = pd.read_csv(file_name, names=names)

        print(f'min_imr = {df["imr_pdr"].min()}, min_pcc = {df["billing_pdr"].min()}')
        if df['imr_pdr'].min() >= 99 and df['billing_pdr'].min() >= 99:
            break
        
        k += 1

    if k == 29:
        k = 28

    _, distances = pairwise_distances_argmin_min(sm_coords, clf.cluster_centers_)
    average_distances_per_cluster = []

    for i in range(clf.n_clusters):
        cluster_distances = distances[clf.labels_ == i]
        average_distance = np.mean(cluster_distances)
        average_distances_per_cluster.append(average_distance)

    file_name = make_file_name(path, k, 'k', sfa, tx_mode, adr_enabled, adr_name)
    pd.DataFrame([k]).to_csv(file_name, header=False, index=False)

    file_name = make_file_name(path, k, 'final_dap', sfa, tx_mode, adr_enabled, adr_name)
    pd.DataFrame(clf.cluster_centers_).to_csv(file_name, header=False, index=False)

    file_name = make_file_name(path, k, 'final_labels', sfa, tx_mode, adr_enabled, adr_name)
    pd.DataFrame(clf.labels_).to_csv(file_name, header=False, index=False)

    file_name = make_file_name(path, k, 'size_clusters', sfa, tx_mode, adr_enabled, adr_name)
    pd.DataFrame(np.bincount(clf.labels_)).to_csv(file_name, header=False, index=False)

    file_name = make_file_name(path, k, 'avg_dists', sfa, tx_mode, adr_enabled, adr_name)
    pd.DataFrame(average_distances_per_cluster).to_csv(file_name, header=False, index=False)

    return k, clf.cluster_centers_, clf.labels_, path

if __name__ == "__main__":
  radius = 7500
  script = 'jisa_2025.cc'
  
  '''nDaps = [1, 1, 1, 1, 1]
  for i, sms in enumerate([200, 400, 600, 800, 1000]):
    sfa = "drsfa"
    path = f'{result_path}/{sfa}/{sms}'
    sm_coords = gen_sm_coords(sms, radius)
    
    sm_file = f'{path}/{sms}sm_file.csv'
    pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)

    km_sbrc_nack(script, sm_coords, sm_file, sfa, path, radius, k=nDaps[i])'''

  nDaps = [14]
  for i, sms in enumerate([200]):
    adr = 'ca_adr'
    sfa = 'adr'
    # path = f'{result_path}/{adr}_{sfa}/{sms}'
    path = f'{result_path}/{adr}/{sms}'
    sm_coords = gen_sm_coords(sms, radius)
    
    sm_file = f'{path}/{sms}sm_file.csv'
    pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)

    km_sbrc_nack(script, sm_coords, sm_file, sfa, path, radius, adr_enabled=1, 
                 adr_name=adr, adr_type="ns3::CAADR", k=nDaps[i])
  
  '''nDaps = [1, 1, 1, 1, 1]
  for i, sms in enumerate([200, 400, 600, 800, 1000]):
    sfa = "isfa"
    path = f'{result_path}/adr_{sfa}/{sms}'
    sm_coords = gen_sm_coords(sms, radius)
    
    sm_file = f'{path}/{sms}sm_file.csv'
    pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)

    km_sbrc_nack(script, sm_coords, sm_file, sfa, path, radius, adr_enabled=1, k=nDaps[i])'''
  
  '''nDaps = [1, 1, 1, 1, 1]
  for i, sms in enumerate([200, 400, 600, 800, 1000]):
    sfa = "adr"
    path = f'{result_path}/adr_{sfa}/{sms}'
    sm_coords = gen_sm_coords(sms, radius)
    
    sm_file = f'{path}/{sms}sm_file.csv'
    pd.DataFrame(sm_coords).to_csv(sm_file, header=False, index=False)

    km_sbrc_nack(script, sm_coords, sm_file, sfa, path, radius, adr_enabled=1, k=nDaps[i])'''