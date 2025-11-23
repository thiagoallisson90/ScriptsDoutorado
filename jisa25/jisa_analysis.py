import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def ic_analysis(data):
  erro_padrao = round(stats.sem(data, ddof=1), 2)
  mean = round(np.mean(data), 2)

  intervalo_conf = stats.t.interval(0.95, len(data)-1, loc=mean, scale=erro_padrao)
  margem_erro = round((intervalo_conf[1] - intervalo_conf[0]) / 2, 2)

  return {
    'mean': float(f'{mean:.2f}'),
    'std_error': float(f'{erro_padrao:.2f}'),
    'ic': (round(intervalo_conf[0], 2), round(intervalo_conf[1], 2)),
    'margin': float(f'{margem_erro:.2f}')
  }

def plot_bars(y1, y2, y3, y4, y5, xlabel, ylabel, xticks, labels, filename, format, width=0.18):
  colors = ['green', 'red', 'blue', 'orange', 'purple']  # 5 cores
  hatches = ['//', '\\', '-', 'x', 'o']                 # Marcadores diferentes
  fontsize = 26
  legendsize = 18

  plt.figure(figsize=(10, 6))
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)

  indices = np.arange(len(y1))
  bars1 = plt.bar(indices - 2 * width, y1, width, label=labels[0], color=colors[0], hatch=hatches[0])
  bars2 = plt.bar(indices - width,      y2, width, label=labels[1], color=colors[1], hatch=hatches[1])
  bars3 = plt.bar(indices,              y3, width, label=labels[2], color=colors[2], hatch=hatches[2])
  bars4 = plt.bar(indices + width,      y4, width, label=labels[3], color=colors[3], hatch=hatches[3])
  bars5 = plt.bar(indices + 2 * width,  y5, width, label=labels[4], color=colors[4], hatch=hatches[4])

  # Função auxiliar para adicionar texto no topo
  def add_labels(bars):
      for bar in bars:
          height = bar.get_height()
          plt.text(
              bar.get_x() + bar.get_width() / 2,
              height,
              f'{height:.0f}',
              ha='center',
              va='bottom',
              fontsize=12,
              fontweight='bold'
          )

  add_labels(bars1)
  add_labels(bars2)
  add_labels(bars3)
  add_labels(bars4)
  add_labels(bars5)

  plt.xticks(indices, xticks, fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)

  # Legenda com borda destacada
  legend = plt.legend(
      ncol=3,
      fontsize=legendsize,
      loc='lower center',
      bbox_to_anchor=(0.5, 1.0),
      frameon=True,
      edgecolor='black',
      handlelength=1.5,
      handleheight=1.2,
      borderpad=1.2
  )

  plt.savefig(f'{filename}.{format}', bbox_inches='tight', dpi=300, format=format)
  plt.close()

def plot_bars1(y1, y2, y3, y4, y5, xlabel, ylabel, xticks, labels, filename, format, width=0.15):
  colors = ['green', 'red', 'blue', 'orange', 'purple']  # 5 cores
  hatches = ['//', '\\', '-', 'x', 'o']                 # Marcadores diferentes
  fontsize = 26
  legendsize = 18

  plt.figure(figsize=(10, 6))
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)

  indices = np.arange(len(y1))
  plt.bar(indices - 2 * width, y1, width, label=labels[0], color=colors[0], hatch=hatches[0])
  plt.bar(indices - width,      y2, width, label=labels[1], color=colors[1], hatch=hatches[1])
  plt.bar(indices,              y3, width, label=labels[2], color=colors[2], hatch=hatches[2])
  plt.bar(indices + width,      y4, width, label=labels[3], color=colors[3], hatch=hatches[3])
  plt.bar(indices + 2 * width,  y5, width, label=labels[4], color=colors[4], hatch=hatches[4])

  plt.xticks(indices, xticks, fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  # plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
  plt.grid(True, linestyle='--')

  # Legenda com borda destacada
  plt.legend(
      ncol=3,
      fontsize=legendsize,
      loc='lower center',
      bbox_to_anchor=(0.5, 1.0),
      frameon=True,
      edgecolor='black',
      handlelength=1.5,
      handleheight=1.2,
      borderpad=1.2
  )

  plt.savefig(f'{filename}.{format}', bbox_inches='tight', dpi=300, format=format)
  plt.close()


def calculate_sem(data):
  return stats.sem(data, axis=1, ddof=1)

def plot_overlapping_bars(data, n, xlabels, xlabel, ylabel, schemes, figname, format):
  cmap = plt.cm.jet
  # cmap = plt.cm.viridis
  # cmap = plt.cm.plasma

  # 6 cores entre 0 e 1 ou 7 cores entre 0 e 1
  cores = None
  hatches = None
  if n == 6:
    #cores = [cmap(i) for i in np.linspace(1, 0, 6)]
    cores = ['blue', '#008080', 'green', 'orange', 'maroon', 'red']
    hatches = ['/', '\\', '|', '-', '+', 'x']
  if n == 7:
    cores = [
      'cyan',
      'blue',
      '#008080',  # teal (intermediária azul-verde)
      'green',
      'orange',
      'maroon',
      'red'
    ]  
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o']
    
  fontsize = 26
  legendsize = 18

  data = np.array(data).T
  x = np.arange(data.shape[1])

  # Cria figura
  plt.figure(figsize=(10, 6))
  plt.ylim(0, 100)

  # Gráfico de barras empilhadas
  bottom = np.zeros_like(x, dtype=float)
  for i in range(n):
      plt.bar(x, data[i], bottom=bottom, label=xlabels[i], color=cores[i], hatch=hatches[i])
      bottom += data[i]

  # Eixos e estilos
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)
  plt.xticks(x, schemes, fontsize=fontsize-4)
  plt.yticks(fontsize=fontsize)
  
  plt.legend(
      ncol=3 if n == 6 else 4,
      fontsize=legendsize,
      loc='lower center',
      bbox_to_anchor=(0.5, 1.0),
      frameon=True,
      edgecolor='black',
      handlelength=1.5,
      handleheight=1.2,
      borderpad=1.2
  )
  
  # plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
  plt.grid(True, linestyle='--', zorder=0)
  # plt.tight_layout()
  plt.savefig(f'{figname}.{format}', bbox_inches='tight', dpi=300, format=format)

def plot_curves_with_error(y1, y2, y3, y4, y5, xlabel, ylabel, xticks, labels, filename, format):
    yerr1 = calculate_sem(y1)
    yerr2 = calculate_sem(y2)
    yerr3 = calculate_sem(y3)
    yerr4 = calculate_sem(y4)
    yerr5 = calculate_sem(y5)

    indices = np.arange(len(y1))

    colors = ['green', 'red', 'blue', 'orange', 'purple']
    markers = ['o', 's', '^', 'd', 'v']  # v = triângulo para baixo
    fmts = ['-', '-.', ':', '-.', '--']

    plt.figure(figsize=(10, 6))

    curves = [(y1, yerr1), (y2, yerr2), (y3, yerr3), (y4, yerr4), (y5, yerr5)]
    for i, (y, yerr) in enumerate(curves):
        plt.errorbar(
            indices, np.mean(y, axis=1), yerr=yerr, fmt=fmts[i],
            marker=markers[i], markeredgecolor='black', markersize=10,
            markeredgewidth=0.8, color=colors[i], label=labels[i],
            capsize=15, capthick=5, elinewidth=5, zorder=2 if i == 0 else 1, 
            linewidth=2
        )

    fontsize = 26
    legendsize = 18

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(indices, xticks, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.grid(True, linestyle='--', linewidth=0.5, zorder=0)
    plt.grid(True, linestyle='--', zorder=0)

    plt.legend(
        ncol=3,
        fontsize=legendsize,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        edgecolor='black',
        handlelength=1.5,
        handleheight=1.2,
        borderpad=1.2
    )

    plt.savefig(f'{filename}.{format}', bbox_inches='tight', dpi=300, format=format)
    plt.close()

def calc_capex(n):
  return 7.1 * n

def make_file_name(base_path, folder, file):
  return f'{base_path}/{folder}/{file}.csv'

if __name__ == '__main__':
  result_path = '/home/thiago/Doutorado/Jisa2025'
  img_path = f'{result_path}/imgs'

  sms = [200, 400, 600, 800, 1000]
  schemes = ['DR-SFA (2-4)', 'I-SFA (2-5)', 'I-SFA+ADR (9-10)', 'CA-ADR (15-28)', 'ADR (28)']
  schemes_min = ['DR-SFA (2-4)', 'I-SFA (2-4)', 'I-SFA+ADR (2-4)', 
                 'CA-ADR (2-4)', 'ADR (2-4)']
  folders = ['drsfa', 'isfa', 'adr_isfa', 'ca_adr_adr', 'adr_adr']

  numbers_of_daps = [[2, 3, 3, 3, 4], [2, 3, 4, 4, 5], [10, 9, 10, 9, 9],
                     [15, 17, 22, 28, 28], [28, 28, 28, 28, 28]]  
  min_daps = [2, 3, 3, 3, 4]

  # Plotar Número de DAPs (barras)
  plot_bars(numbers_of_daps[0], numbers_of_daps[1], numbers_of_daps[2], numbers_of_daps[3],
            numbers_of_daps[4], 'Number of SMs', 'Number of DAPs', ['200', '400', '600', '800', '1000'],
            schemes, f'{result_path}/imgs/daps', 'png')
  '''plot_bars(numbers_of_daps[0], numbers_of_daps[1], numbers_of_daps[2], numbers_of_daps[3],
            numbers_of_daps[4], 'Number of SMs', 'Number of DAPs', ['200', '400', '600', '800', '1000'],
            schemes, f'{result_path}/imgs/daps', 'pdf')'''
  
  # Plotar CAPEX (barras)
  capex = []
  for numbers in numbers_of_daps:
    values = []

    for n in numbers:
      values.append(calc_capex(n))

    capex.append(values)
  
  plot_bars1(capex[0], capex[1], capex[2], capex[3], capex[4], 'Number of SMs', 'CAPEX (k€)', 
             ['200', '400', '600', '800', '1000'], schemes, f'{result_path}/imgs/capex', 'png')
  '''plot_bars1(capex[0], capex[1], capex[2], capex[3], capex[4], 'Number of SMs', 'CAPEX (k€)', 
             ['200', '400', '600', '800', '1000'], schemes, f'{result_path}/imgs/capex', 'pdf')'''

  # Plotar Distância Média (barras)
  names = ['dist']
  dists = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'dists'), names=names)
      l.append(df[names[0]].mean())

    dists.append(l.copy())
    l.clear()
  
  plot_bars1(dists[0], dists[1], dists[2], dists[3], dists[4], 'Number of SMs', 'Avg. Distances (m)', 
             sms, schemes, f'{result_path}/imgs/dist', 'png')
  
  # Plotar PDR (curva)
  names.clear()
  names = ['nSent','nRec','pdr','nImrSent','nImrRec','imrPdr','nPccSent','nPccRec','pccPdr','avgDelay','delayImr',
           'delayPcc','avgRssi','avgSnr','energyCons','tput','ee1','ee2','ee3','ee4','avgPktsRssi','avgPktsSnr','nRun']
  
  pdr = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['pdr'])

    pdr.append(l.copy())
    l.clear()
  
  plot_curves_with_error(pdr[0], pdr[1], pdr[2], pdr[3], pdr[4], 'Number of SMs', 'PDR (%)', 
                         sms, schemes, f'{result_path}/imgs/pdr', 'png')

  print('[308] Estatísticas de PDR: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(pdr):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)
  
  # Plotar PDR para IMR (curva); data => dados para cada esquema, data1 => dados com num de DAPs de DR-SFA
  pdr_imr = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['imrPdr'])

    pdr_imr.append(l.copy())
    l.clear()
  
  plot_curves_with_error(pdr_imr[0], pdr_imr[1], pdr_imr[2], pdr_imr[3], pdr_imr[4], 'Number of SMs', 'PDR (%)', 
                         sms, schemes, f'{result_path}/imgs/pdr_imr', 'png')
  
  print('[331] Estatísticas de PDR para IMR: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(pdr_imr):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)
  
  pdr_imr_min = []
  l = []
  for folder in folders:
    for i in sms:
      file = 'data1'
      if folder == 'drsfa':
        file = 'data'
      elif folder == 'isfa' and i <= 400:
        file = 'data'

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', file), names=names)
      l.append(df['imrPdr'])

    pdr_imr_min.append(l.copy())
    l.clear()
  
  plot_curves_with_error(pdr_imr_min[0], pdr_imr_min[1], pdr_imr_min[2], pdr_imr_min[3], pdr_imr_min[4], 'Number of SMs', 'PDR (%)', 
                         sms, schemes_min, f'{result_path}/imgs/pdr_imr_min', 'png')

  # Plotar PDR para PCC (curva)
  pdr_pcc = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['pccPdr'])

    pdr_pcc.append(l.copy())
    l.clear()
  
  plot_curves_with_error(pdr_pcc[0], pdr_pcc[1], pdr_pcc[2], pdr_pcc[3], pdr_pcc[4], 'Number of SMs', 'PDR (%)', 
                         sms, schemes, f'{result_path}/imgs/pdr_pcc', 'png')
  
  print('[373] Estatísticas de PDR para PCC: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(pdr_pcc):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)
  
  pdr_pcc_min = []
  l = []
  for folder in folders:
    for i in sms:
      file = 'data1'
      if folder == 'drsfa':
        file = 'data'
      elif folder == 'isfa' and i <= 400:
        file = 'data'

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', file), names=names)
      l.append(df['pccPdr'])

    pdr_pcc_min.append(l.copy())
    l.clear()
  
  plot_curves_with_error(pdr_pcc_min[0], pdr_pcc_min[1], pdr_pcc_min[2], pdr_pcc_min[3], pdr_pcc_min[4], 'Number of SMs', 'PDR (%)', 
                         sms, schemes_min, f'{result_path}/imgs/pdr_pcc_min', 'png')

  # Plotar Delay (curva)
  delay = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['avgDelay'])

    delay.append(l.copy())
    l.clear()
  
  plot_curves_with_error(delay[0], delay[1], delay[2], delay[3], delay[4], 'Number of SMs', 
                         'Delay (ms)', sms, schemes, f'{result_path}/imgs/delay', 'png')
  
  print('[415] Estatísticas de Delay: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(delay):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)

  # Plotar Delay para IMR (curva)
  delay_imr = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['delayImr'])

    delay_imr.append(l.copy())
    l.clear()
  
  plot_curves_with_error(delay_imr[0], delay_imr[1], delay_imr[2], delay_imr[3], delay_imr[4], 'Number of SMs', 
                         'Delay (ms)', sms, schemes, f'{result_path}/imgs/delay_imr', 'png')
  
  delay_imr_min = []
  l = []
  for folder in folders:
    for i in sms:
      file = 'data1'
      if folder == 'drsfa':
        file = 'data'
      elif folder == 'isfa' and i <= 400:
        file = 'data'

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', file), names=names)
      l.append(df['delayImr'])

    delay_imr_min.append(l.copy())
    l.clear()
  
  plot_curves_with_error(delay_imr_min[0], delay_imr_min[1], delay_imr_min[2], delay_imr_min[3], delay_imr_min[4], 'Number of SMs', 
                         'Delay (ms)', sms, schemes_min, f'{result_path}/imgs/delay_imr_min', 'png')
  
  # Plotar Delay para PCC (curva)
  delay_pcc = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['delayPcc'])

    delay_pcc.append(l.copy())
    l.clear()
  
  plot_curves_with_error(delay_pcc[0], delay_pcc[1], delay_pcc[2], delay_pcc[3], delay_pcc[4], 'Number of SMs', 
                         'Delay (ms)', sms, schemes, f'{result_path}/imgs/delay_pcc', 'png')
  
  print('[471] Estatísticas de Delay para PCC: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(delay_pcc):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)

  # Plotar SNR para esquemas (curva)
  snr = []
  l = []
  for folder in folders:
    for i in sms:
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', 'data'), names=names)
      l.append(df['avgSnr'])

    snr.append(l.copy())
    l.clear()
  
  plot_curves_with_error(snr[0], snr[1], snr[2], snr[3], snr[4], 'Number of SMs', 
                         'SNR (dB)', sms, schemes, f'{result_path}/imgs/snr', 'png')
  
  print('[494] Estatísticas de SNR: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(snr):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)
  
  snr_min = []
  l = []
  for folder in folders:
    for i in sms:
      file = 'data1'
      if folder == 'drsfa':
        file = 'data'
      elif folder == 'isfa' and i <= 400:
        file = 'data'

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{i}', file), names=names)
      l.append(df['avgSnr'])

    snr_min.append(l.copy())
    l.clear()
  
  plot_curves_with_error(snr_min[0], snr_min[1], snr_min[2], snr_min[3], snr_min[4], 'Number of SMs', 
                         'SNR (dB)', sms, schemes_min, f'{result_path}/imgs/snr_min', 'png')

  # Plotar PLR-I (curva); loss => perdas para cada esquema, loss1 => perdas com num de DAPs de DR-SFA
  names.clear()
  names = ['nInterf','nUnder','nNoMore','nBusy','nExp','nLost','interf_rate','under_rate','nomore_rate','busy_rate',
           'exp_rate','interf_sf7','interf_sf8','interf_sf9','interf_sf10','interf_sf11','interf_sf12','nRun']
  
  interf = []
  l = []
  for j, folder in enumerate(folders):
    for i, n in enumerate(sms):
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', 'loss'), names=names)
      div = numbers_of_daps[j][i]
      l.append(df['nInterf'] / div)

    interf.append(l.copy())
    l.clear()
  
  plot_curves_with_error(interf[0], interf[1], interf[2], interf[3], interf[4], 'Number of SMs', 
                         'Interference', sms, schemes, f'{result_path}/imgs/interf', 'png')
  
  print('[541] Estatísticas de Interferência: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(interf):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)

  interf_min = []
  l = []
  div = 2

  for j, folder in enumerate(folders):
    for i, n in enumerate(sms):
      file = 'loss1'
      if folder == 'drsfa':
        file = 'loss'
      elif folder == 'isfa' and i <= 400:
        file = 'loss'

      if n == 200:
        div = 2
      elif n == 400:
        div = 3
      elif n == 600:
        div = 3
      elif n == 800:
        div = 3
      elif n == 1000:
        div = 4

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', file), names=names)
      l.append(df['nInterf'] / div)

    interf_min.append(l.copy())
    l.clear()

  plot_curves_with_error(interf_min[0], interf_min[1], interf_min[2], interf_min[3], interf_min[4], 'Number of SMs', 
                         'Interference', sms, schemes_min, f'{result_path}/imgs/interf_min', 'png')

  # Plotar PLR-U (curva)
  under = []
  l = []
  for j, folder in enumerate(folders):
    for i, n in enumerate(sms):
      df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', 'loss'), names=names)
      div = numbers_of_daps[j][i]
      l.append(df['nUnder'] / div)

    under.append(l.copy())
    l.clear()
  
  plot_curves_with_error(under[0], under[1], under[2], under[3], under[4], 'Number of SMs', 
                         'Under Sensitivity', sms, schemes, f'{result_path}/imgs/under', 'png')
  
  print('[597] Estatísticas de Under Sensitivity: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(under):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'{sms[j]} SMs')      
      print(ic_analysis(value))
  print('#'*100)

  under_min = []
  l = []
  div = 2

  for j, folder in enumerate(folders):
    for i, n in enumerate(sms):
      file = 'loss1'
      if folder == 'drsfa':
        file = 'loss'
      elif folder == 'isfa' and i <= 400:
        file = 'loss'

      if n == 200:
        div = 2
      elif n == 400:
        div = 3
      elif n == 600:
        div = 3
      elif n == 800:
        div = 3
      elif n == 1000:
        div = 4

      df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', file), names=names)
      l.append(df['nUnder'] / div)

    under_min.append(l.copy())
    l.clear()
  
  plot_curves_with_error(under_min[0], under_min[1], under_min[2], under_min[3], under_min[4], 'Number of SMs', 
                         'Under Sensitivity', sms, schemes_min, f'{result_path}/imgs/under_min', 'png')

  # Plotar Distribuição de SF e Distribuição de TP para 600 SMs (barras sobrepostas)
  names.clear()
  names = ['SF7','SF8','SF9','SF10','SF11','SF12','TP2','TP4',
           'TP6','TP8','TP10','TP12','TP14','nRun']
  sms = [200, 400, 600, 800, 1000]
  
  sf_dist = []
  tp_dist = []
  sf600 = []
  tp600 = []

  l = []
  l1 = []
  m = []
  m1 = []

  for folder in folders:
    n = 600
    df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', 'sf_tp'), names=names)

    for i in range(7, 13):
      l.append(df[f'SF{i}'].mean())
      l1.append(df[f'SF{i}'])
    
    sf_dist.append(l.copy())
    sf600.append(l1.copy())
    
    l.clear()
    l1.clear()

    for i in range(2, 15, 2):
      m.append(df[f'TP{i}'].mean())
      m1.append(df[f'TP{i}'])
    
    tp_dist.append(m.copy())
    tp600.append(m1.copy())
    
    m.clear()
    m1.clear()
  
  plot_overlapping_bars(sf_dist, 6, [f'SF{i}' for i in range(7, 13)], 'Schemes', 'SF Distribution (%)', 
                        ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR'], f'{result_path}/imgs/sf_dist_600', 'png')
  plot_overlapping_bars(tp_dist, 7, [f'{i}' for i in range(2, 15, 2)], 'Schemes', 'TP Distribution (%)', 
                        ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR'], f'{result_path}/imgs/tp_dist_600', 'png')
  
  print('[683] Estatísticas para SF no Cenário com 600 SMs: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(sf600):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'SF{j+7}')   
      print(ic_analysis(value))
  print('#'*100)

  print('[692] Estatísticas para TP no Cenário com 600 SMs: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(tp600):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'TP{(j+1)*2}')      
      print(ic_analysis(value))
  print('#'*100)
  
  sf_dist.clear()
  tp_dist.clear()

  sf600.clear()
  tp600.clear()

  # Plotar Distribuição de SF e Distribuição de TP para 1000 SMs (barras sobrepostas)
  sf1000 = []
  tp1000 = []

  for folder in folders:
    n = 1000
    df = pd.read_csv(make_file_name(result_path, f'{folder}/{n}', 'sf_tp'), names=names)

    for i in range(7, 13):
      l.append(df[f'SF{i}'].mean())
      l1.append(df[f'SF{i}'])

    sf_dist.append(l.copy())
    sf1000.append(l1.copy())

    l.clear()
    l1.clear()

    for i in range(2, 15, 2):
      m.append(df[f'TP{i}'].mean())
      m1.append(df[f'TP{i}'])

    tp_dist.append(m.copy())
    tp1000.append(m1.copy())

    m.clear()
    m1.clear()
  
  plot_overlapping_bars(sf_dist, 6, [f'SF{i}' for i in range(7, 13)], 'Schemes', 'SF Distribution (%)', 
                        ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR'], f'{result_path}/imgs/sf_dist_1000', 'png')
  plot_overlapping_bars(tp_dist, 7, [f'{i}' for i in range(2, 15, 2)], 'Schemes', 'TP Distribution (%)', 
                        ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR'], f'{result_path}/imgs/tp_dist_1000', 'png')
  
  print('[740] Estatísticas para SF no Cenário com 1000 SMs: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(sf1000):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'SF{j+7}')   
      print(ic_analysis(value))
  print('#'*100)

  print('[749] Estatísticas para TP no Cenário com 1000 SMs: ')
  scheme_names = ['DR-SFA', 'I-SFA', 'I-SFA+ADR', 'CA-ADR', 'ADR']
  for i, values in enumerate(tp1000):
    print(scheme_names[i])
    for j, value in enumerate(values):
      print(f'TP{(j+1)*2}')      
      print(ic_analysis(value))
  print('#'*100)

  sf_dist.clear()
  tp_dist.clear()

  sf1000.clear()
  tp1000.clear()

  # Limpar dados
  sms.clear()
  schemes.clear()
  folders.clear()
  
  numbers_of_daps.clear()
  min_daps.clear()

  names.clear()
  dists.clear()

