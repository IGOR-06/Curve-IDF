#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:35:19 2019
@author: igor/eric/henrique
"""

import PySimpleGUI as sg
##############################
import os.path
import PIL.Image
import io
import base64
################################
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
#################################
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import matplotlib
# matplotlib.use('TkAgg')
import pandas as pd
#### importando datetime
import datetime
from datetime import datetime
import scipy as sc
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from matplotlib.ticker import FormatStrFormatter
import statistics as stats

import time

from IPython import get_ipython
try:
    get_ipython().magic("matplotlib","inline")
except:
    plt.ion()
# =============================================================================
# # -----   Escolha do Tema - Green & tan color scheme      ---------------
# =============================================================================
sg.theme('DefaultNoMoreNagging')

# =============================================================================
# # --------       Posição do Texto  ---------------------------
# =============================================================================
sg.set_options(text_justification='left')      

# =============================================================================
# # ---------------    Ajuste do primeiro layout -------------------------
# =============================================================================
items_chosen = [[sg.Text('     Precipitação - Relação Intensidade-Duração-Frequência (IDF)', font=('Times New Roman', 12))],

# =============================================================================
# # -------------      Carregar arquivo excel   -------------------------
# =============================================================================
            [sg.Button('Carregar Arquivo', enable_events=True, key='-READ-', font=('Times New Roman', 9)),
            sg.Input(key='-loaded-', size=(148,1),background_color='white', disabled=True)],           
            
# =============================================================================
# # ------------------  Espaços para entrada de valores  ---------------------
# =============================================================================
           [sg.Text('Nome da Rodada:', font=('Times New Roman', 9), size=(15, 1)), 
            sg.In(default_text='teste1', size=(20, 1),background_color='white',key='nome'), 
            sg.Text('Tempo de decorrelação:', font=('Times New Roman', 9), size=(25, 1)),
            sg.In(default_text='0', size=(5, 1),background_color='white', key='decorrelacao',disabled = True),
            sg.Text('dias', font=('Times New Roman', 9), size=(5, 1)),
            sg.Text('Tempo total em:', font=('Times New Roman', 9), size=(15, 1)),
            sg.Drop(values=('anos', 'meses','dias'), default_value=('anos'), auto_size_text=True, enable_events =True, key='combo'),
            sg.In(default_text='0', size=(8, 1),background_color='white',key='tempo_total',disabled = True),
            sg.Text('Numero amostral:', font=('Times New Roman', 9), size=(16, 1)),
            sg.In(default_text='0000', size=(7, 1),background_color='white', key='n_amostral',disabled = True)],]
          
# =============================================================================
# # -------------------- Inserir Grafico 1  -----------------------            
# =============================================================================

fig = matplotlib.figure.Figure(figsize=(5.6, 3.5), dpi=100)
plt.style.use('ggplot')
ax = fig.add_subplot(111)
ax.axis('off')

# ------------------------------- END OF YOUR MATPLOTLIB CODE -------------------------------

# ------------------------------- Beginning of Matplotlib helper code -----------------------

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

graph1 = [[sg.Canvas(key='-CANVAS-')]]      

# =============================================================================
# # -------------------- Inserir Grafico 2  -----------------------            
# =============================================================================
fig2 = matplotlib.figure.Figure(figsize=(5.6, 3.5), dpi=100)
#t2 = np.arange(0, 3, 1)
# fig2.add_subplot(111).plot(t2, 2 * np.sin(2 * np.pi * t2))
ax = fig2.add_subplot(111)
# fig2.add_subplot(111).plot()
ax.axis('off')
          
graph2 = [[sg.Canvas(key='-CANVAS2-')]]

# =============================================================================
# # -------------------- Colunas de dados ------------------------------
# =============================================================================
options = [[sg.T('Interv. de Seleção de Máximos', font=('Times New Roman', 9)), 
                      sg.In(default_text='0',size=(4,1),background_color='white',
                            key='sel_max',enable_events = True, tooltip = 'Uma alteração pode levar 30 s ou +'), 
                      sg.T('dias', font=('Times New Roman', 9)),
                      sg.Button('Ok', enable_events=True, key='retry', font=('Times New Roman', 9)),
                      sg.Checkbox('Máximos Independentes', font=('Times New Roman', 9),
                                  size=(22, 1), default=False,enable_events=True,
                                  key='max_ind'),
],
          [sg.Frame('Períodos de Retorno',
                    [[sg.T('Recorrência', font=('Times New Roman', 9)), 
                      sg.In(default_text=('1 10 20 30 50 100 1000'), size=(23,1),
                            font=('Times New Roman', 9),background_color='white',
                            key='period_retorn'), 
                      sg.T('anos', font=('Times New Roman', 9))],
                    [sg.T('Número de registros por ano:', font=('Times New Roman', 9)), 
                     sg.In(default_text='0', size=(5,1),background_color='white',
                           key='por_ano',disabled = True), ]  ]),
            sg.Frame('Limiar',
                      [[sg.T('Limiar inferior(mm):', font=('Times New Roman', 9), visible=False), 
                         sg.I(default_text='0',size=(3, 1),background_color='white',
                              key='limiar',enable_events = True, visible=False),
                         sg.I(default_text='100%', size=(5, 1),background_color='white',
                              key='percentual',disabled = True, visible=False)],
                        [sg.T('Número de épocas:', font=('Times New Roman', 9)), 
                         sg.I(default_text='000', size=(5, 1),background_color='white',
                              key='epocas',disabled = True)], ])],
          [sg.T('Análise de Valor Extremo com o ajuste da distribuição:', font=('Times New Roman', 9)),
           sg.Drop(values=('Gumbel', 'Lognormal','Weibull'), default_value=('Gumbel'), auto_size_text=True, enable_events =True, key='distr')],
          [sg.Frame('Relação de Assimetria de Gumbel-Gauss',
                    [[sg.Text('Coef. Angular', font=('Times New Roman', 9), size=(10, 1),key='t_angular'), 
                      sg.Text('Coef. Linear', font=('Times New Roman', 9), size=(9, 1),key='t_linear'),
                      sg.Text('Forma', font=('Times New Roman', 9), size=(10, 1),key='t_pforma',visible=False)
                      ],
                    [sg.Spin(values = ['{:.4f}'.format(i+0.7797) for i in np.arange(-1.5, 1.5, 0.001)], initial_value ='{:.4f}'.format(0.7797),
                             size=(10, 1), background_color='white', key='angular', tooltip = 'Clique e segure para acelerar'),      
                     sg.Spin(values = ['{:.3f}'.format(i-0.450) for i in np.arange(-1.5, 1.5, 0.010)], initial_value ='{:.3f}'.format(0.450),
                             size=(9, 1), background_color='white', key='linear', tooltip = 'Clique e segure para acelerar'),
                     sg.Spin(values = ['{:.3f}'.format(i) for i in np.arange(-3.5, 3.5, 0.010)], initial_value ='{:.3f}'.format(1.00),
                             size=(9, 1), background_color='white', key='pforma', tooltip = 'Clique e segure para acelerar',visible=False) 
                     ]],key='painel_distr'),
          sg.Button('Ajustar', enable_events=True, key='ajustar', font=('Times New Roman', 9)),
          sg.Frame('Resultados',
                      [[sg.Checkbox('Estimativas de Extremos', font=('Times New Roman', 9),
                                    size=(23, 1), default=False,enable_events=True, key='estimativas_extremos')],
                        [sg.Checkbox('Aderência', font=('Times New Roman', 9), size=(20, 1),
                                     default=False, enable_events=True, key='aderencia_check')]])] ]

col1 = sg.Column([[sg.Frame('Equação de Chuvas Intensas',
                    [[sg.T('Tipo de curva IDF:', font=('Times New Roman', 9)),
                        sg.Drop(values=('exponencial', 'logarítmica natural'), default_value=('logarítmica natural'), size=(17, 1), key='tipo_idf', enable_events =True)],
                     [sg.T('   ', font=('Times New Roman', 9)), ],
                      [sg.T('                                 ', font=('Times New Roman', 9)),
                       sg.In(default_text='m',size=(8,1),background_color='white', key='M',disabled = True,visible=False), ],
                      [sg.T('    ', font=('Times New Roman', 9)), 
                       sg.In(default_text='m_ln',size=(10,1),background_color='white', key='K_ou_m_log_natural',disabled = True), 
                      sg.T('ln(T) ', font=('Times New Roman', 9),key = 'texto_periodo_de_retorno'), 
                      sg.In(default_text='+ K_ln',size=(10,1),background_color='white', key='K_lognatural',disabled = True,visible=True)],
                      [sg.T('I = ____________________________    ', font=('Times New Roman', 9))],
                       [sg.T('                                        ', font=('Times New Roman', 9)),
                        sg.In(default_text='n',size=(8,1),background_color='white', key='n',disabled = True), ],
                       [sg.T('       (t + ', font=('Times New Roman', 9)), 
                        sg.In(default_text='t0',size=(9,1), background_color='white', key='t00',disabled = True), 
                      sg.T(')', font=('Times New Roman', 9)), ],
                       [sg.T('   ', font=('Times New Roman', 9)),], ])]])

col2 = sg.Column([[sg.T('Duração em:', font=('Times New Roman', 9)),
                        sg.Drop(values=('horas', 'minutos'), default_value=('horas'), size=(6, 1), key='drop', enable_events =True)],
                        [sg.T('Escolha de t0:', font=('Times New Roman', 9)),
                         sg.I(default_text='0', size=(9, 1), background_color='white', key='t0'),
                         sg.Button('Atualizar',key='botao_t0', font=('Times New Roman', 9))],
            [sg.Frame('Comparação',
                      [[sg.Checkbox('Estimativas IDF', font=('Times New Roman', 9),
                                    size=(18, 1), default=False,enable_events=True,key='idf')],
                        [sg.Checkbox('Diferença e RMS', font=('Times New Roman', 9),
                                     size=(18, 1), default=False,enable_events=True,key='RMS')]])],
# =============================================================================
# # ----------------   Botões para salvar ou fechar  -----------------
# =============================================================================
          [sg.Input(key='salvamento', visible=False, enable_events=True),
           sg.SaveAs('Salvar', font=('Times New Roman', 9),
                     file_types = (('IDF log files', '*.log'),), 
                     enable_events=True,
                     #default_extension = '.log',
                     key='Salvar'), 
          sg.B('Cancelar', font=('Times New Roman', 9))]])
         # sg.Cancel('Cancelar', font=('Times New Roman', 9))]])
# =============================================================================
# # --------   Estruturar Frames e colocar um frame dentro do outro  ---------
# =============================================================================
options2 = [[col1,col2],]

choices = [[sg.Frame('Critério para Análise de Extremos', 
                     font=('Times New Roman', 9), layout= options, size=(585, 215))]]
choices2 = [[sg.Frame('Relação - Duração Intensidade Duração e Frequência (IDF)',
                      font=('Times New Roman', 9), layout= options2, size=(570, 215))]]
# =============================================================================
# # ----------------   Plotar no layout --------------------------
# =============================================================================

layout = [[sg.Column(items_chosen, element_justification='c')], 
           [sg.T(''),sg.Column(graph1, element_justification='c'),sg.T(''),sg.Column(graph2, element_justification='c')],
           [sg.Column(choices, element_justification='c'),sg.Column(choices2, element_justification='c')]]

# =============================================================================
# # --------------   Nome acima da janela  -------------------------
# =============================================================================
window = sg.Window('Curva IDF', layout , finalize=True, font=("Times New Roman", 10),margins=(0,0),resizable=True)      
          
# =============================================================================
# # -----------  adicionar gráficos na janela   -----------------------
# =============================================================================
figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)

# =============================================================================
# # -----------  funções para selecionar máximos independentes  ----------------
# =============================================================================
def selecao_max_ind(df,DT,**kwargs):
    limiar = kwargs.get('limiar',(np.floor(df.min().min())-1e-9,np.ceil(df.max().max())+1e-9))

    if np.isnan(limiar[0]):
        limiar[0] = np.floor(df.min().min())-1e-9
    elif np.isnan(limiar[1]):
        limiar[1] = np.ceil(df.max().max())+1e-9
    
    lpmax = [None]*len(df.columns) # lista
    dpmax = {}                     # dicionário
    # Loop para a seleção de máximos das diferentes durações de chuva
    dfs = df.resample('1h').asfreq()
    for duracao,j in zip(df.columns,range(len(df.columns))):
        #pmax = seleciona_maximos_independentes(df[duracao].copy(),DT,limiar=limiar)
        index, peaks = find_peaks(dfs[duracao], height=limiar,distance=DT*24)
        pmax = pd.Series(peaks['peak_heights'],index=dfs.index[index])
        pmax.name = duracao # nomeando a pd.Series para utilizar como coluna no DF
        lpmax[j] = pmax
        dpmax[duracao] = pmax
    
#-----------------------------------------------------------------------------

    # DataFrame (DF) com os máximos selecionados
    prec_max = pd.concat(lpmax,axis=1)

    # número de épocas
    nes = prec_max.notna().sum().min()

    # Ordena de forma decrescente pois assim o valores menores ficam por último
    lpre_max = [prec_max[col].sort_values(ascending=False).dropna()[0:nes].reset_index(drop=True) for col in prec_max.columns]
    pre_max = pd.concat(lpre_max, axis=1)

    # 02 dataframes com as épocas - 01 sem índice de tempo  e Outro com os indices de tempo para plotagem
    return pre_max, prec_max


def seleciona_maximos_independentes(serie,DT,**kwargs):
    limiar = kwargs.get('limiar',(np.floor(serie.min())-1e-9,np.ceil(serie.max())+1e-9))
    if np.isnan(limiar[0]):
        limiar[0] = np.floor(serie.min())-1e-9
    elif np.isnan(limiar[1]):
        limiar[1] = np.ceil(serie.max())+1e-9
           
    # Organiza a série em ordem cronológica ascendente
    serie.sort_index(inplace=True)

    # Retirada dos valores que extrapolam os limiares superiores e inferiores
    #serie[(serie<=limiar[0]) | (serie>=limiar[1])] = np.nan # Não funcionou
    # serie[(serie<=limiar[0]) | (serie>=limiar[1])] = np.timedelta64('Nat') # Não funcionou
    serie = serie[(serie>limiar[0]) & (serie<limiar[1])] 

    # cálculo dos picos - Tentativa de otimização
    #sd = np.sign(np.diff(serie))
    #sd = np.append(-sd[0],np.append(sd,-sd[-1]))
    #dsd= np.diff(sd)    

    # Mantém somente onde há picos na série
    #serie = serie[dsd==-2] # -2 == cristas, 2 == cavas

    # ti = tempo inicial  e tf = tempo final para seleção de máximos
    ti = serie.index[0]
    ti = ti.replace(hour=0, minute=0)
    tf = ti + np.timedelta64(int(DT*24*3600),'s')         
    jj=0 
    
    # Criação da variáveis auxliares
    v_max=[] # lista com os máximos
    t_max=[] # lista com os tempos dos máximos
    
    # Primeiro passo do loop while
    k = (serie.index >= ti) & (serie.index < tf)
    if k.any():
        v_max.append(serie[k].max())    # registra o valor máximo
        t_max.append(serie[k].idxmax()) # registra o índice de tempo do valor máximo

    ti = tf
    tf = ti + np.timedelta64(int(DT*24*3600),'s') # Atualiza a janela de busca.

    while ti <= serie.index[-1]:
        k = (serie.index >= ti) & (serie.index < tf)
        if k.any():
            if (serie[k].idxmax() - t_max[jj]) < np.timedelta64(int(DT*24*3600),'s'):
                if serie[k].max() > v_max[jj]:
                    v_max.pop(jj) # retirada do valor para posterior substituição
                    t_max.pop(jj) # retirada do registro de tempo para posterior substituição
                    
                    v_max.append(serie[k].max()) # substituição pelo valor maior
                    t_max.append(serie[k].idxmax()) # substituição pelo tempo associado ao valor maior
            else:
                jj=jj+1
                v_max.append(serie[k].max())
                t_max.append(serie[k].idxmax())
        ti = tf
        tf = ti + np.timedelta64(int(DT*24*3600),'s') # Atualiza a janela de busca.
        
    prec_max = pd.Series(data=v_max,index=t_max) # pandas série de saída
    
    return prec_max


# =============================================================================
# # ---------------  função de cáclulo do tempo de decorrelação  --------------
# =============================================================================
def t_decorr(serie,tolt,**kwargs):
    extensao = kwargs.get('extensao',30) # Default 30 dias para o cálculo
    
    # Intervalo modal da série temporal.
    interv_continuo = stats.mode(np.diff(serie.index))
    
    # número de registros de meia janela.
    nrj = int(np.round(np.timedelta64(int(tolt*24*3600*1e9))/interv_continuo))
    
    #Número de registros correspondentes a extensao dias.
    nreg = int(np.timedelta64(int(extensao*24*3600*1e9),'ns')/ interv_continuo )
    t = [pd.Series.autocorr(serie,lag=x) for x in range(nreg)]

    # diferença  do sinal da 1ª diferença da autocorr.
    dd = np.diff(np.sign(np.diff(t)))
    
    # pontos onde a curva de autocorr. se inflexiona para cima.
    inflex = (dd==2).nonzero()[0]+1
    
    # Encontra o primeiro mínimo local onde a curva começa a oscilar. 
    aonde = 0
    for i in inflex:
        if (t[i] < 0.3) & (t[i] == min( t[max(i-nrj,1):min(i+nrj,len(t))] ) ):
            aonde = i;
            break
    
    # Intervalo amostral
    ia = float(interv_continuo)
    
    # Soma + 1 ao oande pois o indice no python é baseado no zero!!
    tempo_decorr = (aonde+1)*ia/86400e9 # tempo de decorrelação em dias
    
    # tratamento quando o formato de data e tempo for datetime64[ns]
    if np.dtype('<M8[ns]') == serie.index.dtype:
        ia = ia/86400e9/365.25 # intervalo amostral em anos
    
    return tempo_decorr, ia

# =============================================================================
# # -----------  função ler tabela   -----------------------
# =============================================================================
def read_table():
    sg.set_options(auto_size_buttons=True)
    filename = sg.popup_get_file(
        'Dataset to read',
        title='Dataset to read',
        no_window=True, 
        file_types=(("CSV Files", "*.csv"),("Text Files", "*.txt")))
    # --- populate table with file contents --- #
    if filename == '':
        return
    data = []
    header_list = []
    if filename is not None:
        # Nome do arquivo de entrada
        fn = filename #.split('/')[-1]

        # Nome do arquivo de saída
        window.Element('salvamento').update(window.Element('-loaded-').get().split('.')[0] + '.log')
        
        # leitura do arquivo Excel
        df = pd.read_csv(filename, sep=';',skiprows=8, engine='python', encoding='latin-1',
                          parse_dates= [0],dayfirst=True,index_col=0)
        
        header_list = df.columns
        # ---------------------------------------------------------------
        
        # Tempo de decorrelação
        tempo_decorr, ia = t_decorr(df.iloc[:,-1].copy(),1,extensao=20)

        # Só há necessidade de alterar a cada mudança de arquivo
        window.Element('n_amostral').update(df.shape[0]) # numero amostral #
        window.Element('tempo_total').update(value = '{:.2f}'.format(float(len(df.index)*ia) ))
        
        # Texto para ser utilizado nas tabelas de estimativas de extremos.
        tab_index = 'Máx. obs. ('+ "{:.2f}".format(float(len(df.index)*ia)) + ' anos)'

        # Linha adicional para a tabela de estimativa de extremos                                                                            
        vmaxs = pd.DataFrame(df.max().to_dict(),index=[tab_index],columns= df.columns)
        
        info_adi_tab_est = {'tempo_total':float(len(df.index)*ia),
         'valores_maximos':vmaxs}
        
        window.Element('tempo_total').metadata = info_adi_tab_est
        window.Element('decorrelacao').update('{:.2f}'.format(tempo_decorr)) # tempo_decorrelação #
        
        # Primeira estimativa de um intervalo de seleção de epocas
        window.Element('sel_max').update(np.round(tempo_decorr)) # sel. de epocas #
        
        window.Element('-loaded-').metadata = df
        
        return (df, header_list, fn, tempo_decorr, ia)

# =============================================================================
# # -------  funções para  calcular as estimativas de precipitação ------------
# =============================================================================
#++++++++++++++++++++ WEIBULL
def calc_values_precip_weibull(pre_max,T,coefLin,coefAng,**kwargs):
    coefFor=kwargs.get('forma',[])
    # Número de épocas selecionadas da série com o menor número de registros 
    nes = pre_max.shape[0]
  
    # tempo total
    nanos = window['tempo_total'].metadata['tempo_total']
    
    # Número de registros por ano.
    nrpa = nes/nanos

    # DataFrame (DF) dos perídos de retorno
    T = pd.DataFrame(T)
    T.columns=['retorno']
    
    # Probabilidade d não excedência
    probr = 1-(1/(T*nrpa))

    
    # DataFrame de saída das estimativas de extremos
    xt = pd.DataFrame([],index = T['retorno'], columns = pre_max.columns)
    xt = xt.astype(float)

    para_loc = [None]*len(pre_max.columns)
    para_esc = [None]*len(pre_max.columns)
    para_for = [None]*len(pre_max.columns)
    
    # Loop de cálculo
    for i in range(0,len(pre_max.columns)):
        para_for[i], para_loc[i], para_esc[i]  =  sc.stats.weibull_min.fit(pre_max.iloc[:,i],loc=0,scale=1)
        para_loc[i] = coefLin*para_loc[i]
        para_esc[i] = coefAng*para_esc[i]
        para_for[i] = coefFor*para_for[i]
        xt.iloc[:,i] = sc.stats.weibull_min.isf(1-probr, c=para_for[i],loc=para_loc[i],scale=para_esc[i])
    
    # -------------------------------------------------------------------
    # Atualização dos objetos da interface gráfica
    # -------------------------------------------------------------------
    window.Element('por_ano').update('{:.2f}'.format(nrpa)) # numero de registro por ano #
    window.Element('epocas').update(nes) # numero de epocas #
    
    # -------------------------------------------------------------------
    # Armazenamento do DF com as épocas (máimxos no caso da chuva)
    # -------------------------------------------------------------------
    window.Element('epocas').metadata = pre_max
    
    # -------------------------------------------------------------------
    # Armazenamento dasprecitações dos diferentes períodos de retorno
    # -------------------------------------------------------------------
    window.Element('estimativas_extremos').metadata = xt
    
    return xt, nes, nrpa    

#++++++++++++++++++++ LOGNORMAL
def calc_values_precip_lnorma(pre_max,T,coefLin,coefAng, **kwargs):
    coefFor = kwargs.get('forma',[])
    # Número de épocas selecionadas da série com o menor número de registros 
    nes = pre_max.shape[0]
  
    # tempo total
    nanos = window['tempo_total'].metadata['tempo_total']
    
    # Número de registros por ano.
    nrpa = nes/nanos

    # DataFrame (DF) dos perídos de retorno
    T = pd.DataFrame(T)
    T.columns=['retorno']
    
    # Probabilidade d não excedência
    probr = 1-(1/(T*nrpa))


    #media = pre_max.mean()
    #desvio = pre_max.std()
    
    #s_lognor = np.log((pre_max - media)/desvio).std()
    #l_lognor = np.log((pre_max - media)/desvio).mean()
    
    #para_loc = coefLin*pre_max.apply(np.log).mean()
    #para_esc = coefAng*pre_max.apply(np.log).std()
    
    para_loc = [None]*len(pre_max.columns)
    para_esc = [None]*len(pre_max.columns)
    para_for = [None]*len(pre_max.columns)
    
    #desv_pad = pre_max.std()
    
    # DataFrame de saída das estimativas de extremos
    xt = pd.DataFrame([],index = T['retorno'], columns = pre_max.columns)
    xt = xt.astype(float)

    # Loop de cálculo
    for i in range(0,len(pre_max.columns)):
        forma, locacao, escala = sc.stats.lognorm.fit(pre_max.iloc[:,i])
        para_loc[i]=coefLin*locacao
        para_esc[i]=coefAng*escala
        para_for[i]=coefFor*forma
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s=desvio[i], loc = media[i])
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s = s_lognor[i], loc = l_lognor)
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s = s_lognor[i], loc = 0, scale = desvio[i])
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s=s_lognor[i], loc = media[i], scale = desvio[i])
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s=1, loc = media[i], scale = desvio[i])
        xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, para_for[i], loc = para_loc[i], scale = para_esc[i])
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s=1, loc = para_loc[i], scale = para_esc[i])
        #xt.iloc[:,i] = sc.stats.lognorm.isf(1-probr, s=para_esc[i]**2, loc = para_loc[i], scale = 1)
    
    # -------------------------------------------------------------------
    # Atualização dos objetos da interface gráfica
    # -------------------------------------------------------------------
    window.Element('por_ano').update('{:.2f}'.format(nrpa)) # numero de registro por ano #
    window.Element('epocas').update(nes) # numero de epocas #
    
    # -------------------------------------------------------------------
    # Armazenamento do DF com as épocas (máimxos no caso da chuva)
    # -------------------------------------------------------------------
    window.Element('epocas').metadata = pre_max
    
    # -------------------------------------------------------------------
    # Armazenamento dasprecitações dos diferentes períodos de retorno
    # -------------------------------------------------------------------
    window.Element('estimativas_extremos').metadata = xt
    
    return xt, nes, nrpa    

#++++++++++++++++++++ GUMBEL
def calc_values_precip_gumbel(pre_max,T,coefLin,coefAng):

    # Número de épocas selecionadas da série com o menor número de registros 
    nes = pre_max.shape[0]
  
    # tempo total
    nanos = window['tempo_total'].metadata['tempo_total']
    
    # Número de registros por ano.
    nrpa = nes/nanos

    # DataFrame (DF) dos perídos de retorno
    T = pd.DataFrame(T)

    # ------------------------------------------------------------------------
    yt = -np.log(-np.log(1-(1/(T*nrpa))))
    
    # ------------------------------------------------------------------------
    k = (coefAng * yt) - coefLin
    
    # ----------------------------------------------------------------------
    # Ordena de forma decrescente pois assim o valores menores ficam por último
    #pre_max = prec_max.sort_values(by=list(prec_max.columns),axis=0)
    #pre_max=pre_max.iloc[::-1]
    
    # Faz-se a média dos "nes" escolhidos
    media = pre_max.mean().values.tolist()
    desvio= pre_max.std().values.tolist()
    
    desvio2 = pd.DataFrame(desvio) 
    desvio2 = desvio2.T

    # ----------------------------------------------------------------------
    # Estimativa de extremos
    # ----------------------------------------------------------------------
    xt = (media+(k.dot(desvio2)))
    T.columns=['retorno']
    xt.set_index(T['retorno'],inplace=True)
    xt.columns = pre_max.columns
    
    # -------------------------------------------------------------------
    # Atualização dos objetos da interface gráfica
    # -------------------------------------------------------------------
    window.Element('por_ano').update('{:.2f}'.format(nrpa)) # numero de registro por ano #
    window.Element('epocas').update(nes) # numero de epocas #
    
    # -------------------------------------------------------------------
    # Armazenamento do DF com as épocas (máimxos no caso da chuva)
    # -------------------------------------------------------------------
    window.Element('epocas').metadata = pre_max
    
    # -------------------------------------------------------------------
    # Armazenamento dasprecitações dos diferentes períodos de retorno
    # -------------------------------------------------------------------
    window.Element('estimativas_extremos').metadata = xt

    return xt, nes, nrpa
    
# =============================================================================
# # -----------  função calcular as estimativas das curvas IDF ----------------
# =============================================================================
def calc_curvas(xt,**kwargs):
    t0=kwargs.get('t0',[])

    # -------------------------------------------------------------------
    # Definição da intensidade
    # -------------------------------------------------------------------
    duracao = np.array([int(x.replace('h','')) for x in xt.columns.tolist()]) #[1,2,3,4,6,9,12,18,24]
    T2_matrix = pd.DataFrame([duracao] * len(xt.index),index = xt.index, columns = xt.columns)
    intensid = xt/T2_matrix

    # -----------------------------------------------------------------------
    # A unidade de tempo da duração t é sempre em min !  
    # (Capítulo 4 Precipitação, Parte 3: Análise de Chuvas Intensas, Dr. Doalcey Antunes Ramos)
    # -----------------------------------------------------------------------   
    #if window.Element('drop').get() =='minutos':
    if True in ['h' in x for x in xt.columns.tolist()]:
        duracao = duracao*60  
    
    # chute inicial é 40 min
    t_ini = 540
    unidade = 'min'
  
    # -----------------------------------------------------------------------
    # definição de t0
    # -----------------------------------------------------------------------   
    if isinstance(t0,float)==False:
        # otimização para a estimativa de t0
        tzero=[np.nan]*len(intensid)
        #fmint0=[np.nan]*len(intensid) 
        
        for itr in range(intensid.shape[0]):
            ip = intensid.iloc[itr,:].values
            #fmint0 = lambda x: 1-np.abs(pearsonr(np.log10(ip),np.log10(duracao+x))[0])
            fmint0 = lambda x: 1-pearsonr(np.log(ip),np.log(duracao+x))[0]**2
            tzero[itr]= float(sc.optimize.fmin(func=fmint0, x0=t_ini,disp=0))
            del fmint0
        
        t0=float(np.mean(tzero))
    
    # --------------------------------------------------------------------
    # Resolvendo a relação através da aplicação de uma anamorfose logarítmica na equação;
    # i = C/(t+t0)^n = KT^m/(t+t0)
    # --------------------------------------------------------------------
    #breakpoint()
    xp = np.log10(t0 + duracao)
    yp = np.log10(intensid.T)

    # --------------------------------------------------------------------
    # Ajuste polinomial : i = C/(t+t0)^n
    # --------------------------------------------------------------------
    #p = pd.DataFrame(index=xt['retorno'], columns=['a', 'b'])
    p = pd.DataFrame(index=xt.index, columns=['a', 'b'])
    for j in range(0,len(T)):
        p.iloc[j,:] = np.polyfit(xp, yp.iloc[:,j], deg=1)
    
    # Escolha do parâemtro "n" da curva IDF
    p = p.astype(float) 
    n = p.iloc[:,0].mean() # Mediana
    #C = np.exp(p.iloc[:,1]) # valores de C
    C = 10**(p.iloc[:,1]) # valores de C
    
    #breakpoint()
    # ------------------------------------------------------------------------
    # DataFrame com os resultados da função
    # ------------------------------------------------------------------------
    t_func = pd.DataFrame(index=xt.index, columns=duracao)

    if window.Element('tipo_idf').get() == 'exponencial':
        # --------------------------------------------------------------------
        # Resolvendo a relação através da aplicação de uma anamorfose logarítmica na equação;
        # i = KT^m/(t+t0)^n = C/(t+t0)^n
        # --------------------------------------------------------------------
        xp2 = np.log10(xt.index.values)
        yp2 = np.log10(C.values)
        
        # --------------------------------------------------------------------
        # Ajuste polinomial : i = KT^m/(t+t0)^n
        # --------------------------------------------------------------------
        coef = np.polyfit(xp2,yp2, deg=1)
        m = coef[0]
        #K = np.exp(coef[1])
        K = 10**(coef[1])
    
        # Cálculo em duas etapas
        etapa1 = np.array((K * ((t_func.index[:])**m)))

        # Atualização dos objetos da interface gráfica        
        window.Element('K_lognatural').update(visible=False) # coeficiente m #
        window.Element('M').update('{:.4f}'.format(m),visible=True) # coeficiente m #
        window.Element('K_ou_m_log_natural').update('{:.4f}'.format(K)) # coeficiente K #
        window.Element('texto_periodo_de_retorno').update('T')
        
    elif window.Element('tipo_idf').get() == 'logarítmica natural':
        # --------------------------------------------------------------------
        # Resolvendo a relação através da aplicação de uma anamorfose logarítmica na equação;
        # i = (m * ln(T) + K )/(t+t0)^n = C/(t+t0)^n
        # --------------------------------------------------------------------
        xp2 = np.log(xt.index.values)
        yp2 = C.values
        
        # --------------------------------------------------------------------
        # Ajuste polinomial : i = KT^m/(t+t0)^n
        # --------------------------------------------------------------------
        coef = np.polyfit(xp2,yp2, deg=1)
        m = coef[0]
        #K = np.exp(coef[1])
        K = coef[1]

        # Cálculo em duas etapas
        etapa1 = np.array((m * np.log(t_func.index[:]) + K))
        
        # Atualização dos objetos da interface gráfica        
        window.Element('K_lognatural').update('{:+.4f}'.format(K),visible=True) # coeficiente m #
        window.Element('M').update('{:.4f}'.format(m),visible=False) # coeficiente m #
        window.Element('K_ou_m_log_natural').update('{:.4f}'.format(m)) # coeficiente K #
        window.Element('texto_periodo_de_retorno').update('ln(T) ')
        
    # ------------------------------------------------------------------
    # Cálculo em duas etapas
    # ------------------------------------------------------------------
    etapa2 = np.array((t_func.columns[:] + t0)**n)
    
    #breakpoint()
    # calculo da intensidade em função da frequência e duração
    t_func = pd.DataFrame(data = etapa1.reshape(-1,1) * etapa2, index=xt.index, columns=xt.columns) 
    
    # ------------------------------------------------------------------
    # -------------------------------------------------------------------
    diference_inte = t_func - intensid # diferença entre idf e (precipitação Análise de Extremos)/duração 
    
    diference_prec = t_func*duracao - xt # diferença entre idf*duracao e precipitação gumbel
    
    diference = {
        'precipitacao':{'valor':diference_prec,'unidade':'mm'},
        'intensidade' :{'valor':diference_inte,'unidade':'mm/h'}
        }
    # -------------------------------------------------------------------
    # Armazenamento das diferenças
    # -------------------------------------------------------------------
    window.Element('RMS').metadata = diference
                   
    # -------------------------------------------------------------------
    # Atualização dos objetos da interface gráfica
    # -------------------------------------------------------------------
    window.Element('n').update('{:.4f}'.format(-n)) # coeficiente n #
    window.Element('t00').update('{:.4f}'.format(t0)) # coeficiente t0 # Na formula (edição desabilitada)
    window.Element('t0').update('{:.4f}'.format(t0)) # coeficiente t0 # Na caixa de edição habilitada
    
    # -------------------------------------------------------------------
    # Armazenamento das curvas IDF mm/h
    # -------------------------------------------------------------------
    window.Element('idf').metadata = t_func
   
    return t_func, intensid, diference, K, m, t0, n
    
# =============================================================================
# # --------------  função plotar gráfico   -----------------------
# =============================================================================
def plot_fig(pre_max,xt):
    """
    Plots
    """
    # núemro de épocas
    nes = pre_max.shape[0]
    
    # Coeficientes da Relação de Assimetria e/ou Escala, Locação e Forma 
    coefAng = float(window.Element('angular').get())
    coefLin = float(window.Element('linear').get())
    coefFor = float(window.Element('pforma').get())
    
    # estatística básica
    #minx = pre_max.min().min()
    #maxx = xt.max().max()
    if window.Element('distr').get()=='Gumbel':
        media = pre_max.mean().values.tolist()
        desvio= pre_max.std().values.tolist()
        
        # Parâmetros de locação e escala de Gumbel
        para_loc = [i-j*coefLin for i,j in zip(media,desvio)]
        para_esc = [i*coefAng for i in desvio]

    elif window.Element('distr').get()=='Weibull' or window.Element('distr').get()=='Lognormal':
        para_loc = [None]*len(pre_max.columns)
        para_esc = [None]*len(pre_max.columns)
        para_for = [None]*len(pre_max.columns)
        
           
    # número de registros por ano
    nrpa = nes/float(window.Element('tempo_total').get())
    
    # Probabilidade das épocas
    eprob = np.linspace(1/nes, 1-1/nes, nes)
    
    # tempo associado as épocas
    tepocas = 1/(1-eprob)/nrpa
    
    fig = matplotlib.figure.Figure(figsize=(5.6, 3.5), dpi=100)
    # ----------------------------------------------------------------------
    ax = fig.add_subplot(111)
    ax.set_yscale("log")
    ax.set_yticks([ 0, 0.01,  0.1,  1,  10, 20, 30, 50, 100, 1000]) 
    # ax.set_ylim(0, 1000)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
   
    for i in range(0,len(xt.columns)):
        # ax.scatter(y = xt.index, x=xt.values.T[i], label=xt.columns[i], marker='+')#, edgecolor='k')
        p=ax.plot(xt.values.T[i],xt.index, lw=1)#, label=xt.columns[i])
        ax.plot(pre_max.iloc[:,i].sort_values(),tepocas, "+", lw=1, color=p[-1].get_color(), label=xt.columns[i])
        
        if window.Element('distr').get()=='Gumbel':
            ax.plot(sc.stats.gumbel_r.isf(1-eprob,para_loc[i],para_esc[i]),tepocas, lw=1, color=p[-1].get_color())
        
        elif window.Element('distr').get()=='Lognormal':
            # ajuste dos parâmetros
            forma, locacao, escala = sc.stats.lognorm.fit(pre_max.iloc[:,i])
            para_loc[i]=coefLin*locacao
            para_esc[i]=coefAng*escala
            para_for[i]=coefFor*forma
            ax.plot(sc.stats.lognorm.isf(1-eprob,para_for[i],loc=para_loc[i],scale=para_esc[i]),tepocas, lw=1, color=p[-1].get_color())
        
        elif window.Element('distr').get()=='Weibull':
            para_for[i], para_loc[i], para_esc[i]  =  sc.stats.weibull_min.fit(pre_max.iloc[:,i],loc=0,scale=1)
            para_loc[i] = coefLin*para_loc[i]
            para_esc[i] = coefAng*para_esc[i]
            para_for[i] = coefFor*para_for[i]
            ax.plot(sc.stats.weibull_min.isf(1-eprob, c=para_for[i],loc=para_loc[i],scale=para_esc[i]),tepocas, lw=1, color=p[-1].get_color())
 
    font = {'color':'black','weight':'normal','size':8}
    ax.legend(loc='lower right')
    ax.set_ylabel('Periodo de Retorno (anos)',fontdict=font)
    ax.set_xlabel('Extremo de Precipitação (mm)',fontdict=font)
        
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%s'))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    #ax.set_xticks(ticks=range(0,350,50))
    ax.set_title('Probabilidade de '+window.Element('distr').get(), fontdict=font)
    
    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg
              
    return (fig)

# =============================================================================
# # -------------------- Inserir Grafico 2   -----------------------            
# # -------------------- Plot das Curvas IDF -----------------------            
# =============================================================================
def plot_fig2(t_func,xt):
    """
    Plots
    """

    # t0
    t0=float(window.Element('t0').get())
    
    # -------------------------------------------------------------------
    # Definição da intensidade em mm/h conforme o 
    # (Capítulo 4 Precipitação, Parte 3: Análise de Chuvas Intensas, Dr. Doalcey Antunes Ramos)
    # -------------------------------------------------------------------
    # Duração da chuva em horas
    t_h = np.array([float(x.replace('h','')) for x in t_func.columns.tolist()]) 
    # Intensidade a partir das estimativas de gumbel
    intens_tr = xt/t_h # intensidade em mm/h
    #breakpoint()
    # Conversão do tempo conforme a seleção do usuário
    #if True in ['h' in x for x in t_func.columns.tolist()]:
    #    if window.Element('drop').get() != 'horas':
    
        
    # Conversão do tempo de duração da chuva conforme a seleção do usuário
    #if window.Element('drop').get() != 'horas':
    #    yticks=[.001, .01, 1, 10, 50, 100, 200]
    #    xticks=[1,60,240,720,1440,5760]
    #    xticks=[1,2,3,4,6,10,20,30]
        
    #else:

    # -----------------------------------------------------------------------
    # A unidade de tempo da duração t é sempre em min !  
    # (Capítulo 4 Precipitação, Parte 3: Análise de Chuvas Intensas, Dr. Doalcey Antunes Ramos)
    # -----------------------------------------------------------------------   
    t_min = t_h*60 # Passando do tempo de duração em hora para minuto!
    yticks=[1,10,20,30,40,50,60,70,80,90,100,200,300,400,1000]
    xticks=[1,60,240,720,1440,5760]

    fig2 = matplotlib.figure.Figure(figsize=(5.6, 3.5), dpi=100)

    # -----------------------------------------------------------------------
    ax = fig2.add_subplot(111)
    ax.set(xscale='log',yscale='log',
           #ylim=(9,150),
           yticks=yticks,
           #xlim=([1,25]),
           xticks=xticks
           #xticklabels=['','','','','10¹']
           )
   
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for i in range(0,len(t_func.index)):
        hidf = ax.plot(t_min+t0, t_func.values[i], lw=1, label='{:,.0f}'.format(t_func.index[i]) +' ano(s)')
        ax.plot(t_min+t0, intens_tr.values[i],'--', lw=1,color=hidf[-1].get_color())

    font = {'color':'black','weight':'normal','size':8}
    ax.legend(loc='upper right')
    ax.set_ylabel('Precipitação (mm/h)',fontdict=font)
    ax.set_xlabel('Duração (h)',fontdict=font)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title('Relações IDF', fontdict=font)

    return (fig2)
# =============================================================================
# # --------------------   deletar figuras antigas  -----------------------
# =============================================================================
def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')
    
def delete_figure_agg2(figure_agg2):
    figure_agg2.get_tk_widget().forget()
    plt.close('all')
    
# =============================================================================
# # --- Transforma o DF em lista para apresentar como tabela no PySimleGui --- 
# =============================================================================
def df_ext_2_lista(xt,**kwargs):
    vmaxs = kwargs.get('vmaxs',[])
    
    nxt = xt.astype('object')
    nxt = nxt.applymap('{:,.4f}'.format)
    
    nxt =  nxt.append(pd.DataFrame([[' ']*xt.shape[1]],index=[' '],columns=xt.columns))
    if len(vmaxs):
        nxt = nxt.append(vmaxs)
    
    # Talvez o Andrioni otimize o bloco abaixo
    ldados = nxt.values.tolist()
    for x,y in zip(range(len(ldados)),nxt.index.values.tolist()):
        if isinstance(y,float) or isinstance(y,int):
            ldados[x].insert(0,str(int(y))+' anos (mm)')
        elif isinstance(y,str):
            ldados[x].insert(0,y)
                  
    lhead = nxt.columns.values.tolist()
    lhead.insert(0,'Perído de Retorno')
    
    return ldados,lhead

# =============================================================================
# # ---------- Tabela com as estimativas de precipitação da Estatística de Extremos ----------   
# =============================================================================
def tab_estimativa_extremos(xt,**kwargs):

    fn = kwargs.get('titulo',' ')    
    vmaxs = kwargs.get('vmaxs',[])

    ldados,lhead = df_ext_2_lista(xt,vmaxs=vmaxs)

    layout = [
        [sg.Table(values= ldados,
                  headings=lhead,
                  font='Helvetica',
                  def_col_width=45,
                  background_color='lightgrey',
                  alternating_row_color='white',
                  auto_size_columns=True,
                  key = 'wtab_gumbel'
                  )]
    ]

    wtab_estimativa_extremos = sg.Window('Estimativas de '+window.Element('distr').get()+' de Precipitação '+fn, layout, grab_anywhere=False)
    wtab_estimativa_extremos.read(timeout=150)
    
    return wtab_estimativa_extremos

# =============================================================================
# # --------------- Valor crítico do teste de Kolmogorov-Smirnov ---------------
# =============================================================================
def ks_crit_val(alpha1, n,**kwargs):
    # KS Test critical value
    caso = kwargs.get('alternative','two-sided')
    
    if caso=='two-sided':
        alpha1 = alpha1/2 # two-sided
    
    A              =  0.09037*(-np.log10(alpha1))**1.5 + 0.01515*np.log10(alpha1)**2 - 0.08467*alpha1 - 0.11143
    asymptoticStat =  np.sqrt(-0.5*np.log(alpha1)/n)
    criticalValue  =  asymptoticStat - 0.16693/n - A/n**1.5
    criticalValue  =  min(criticalValue , 1 - alpha1)

    return criticalValue

# =============================================================================
# # ----- Testes de aderência do ajuste de Exterme Value Analysis (EVA) ------
# =============================================================================
def aderencia_2_lista(pre_max, coefLin, coefAng, nes, **kwargs):
    coefFor = kwargs.get('forma',[])
    
    # ----------------------------------------------------------------------
    # Ordena de forma decrescente pois assim o valores menores ficam por último
    #lpre_max = [prec_max[col].sort_values(ascending=False).dropna()[0:nes].reset_index(drop=True) for col in prec_max.columns]
    #pre_max = pd.concat(lpre_max, axis=1)

    # Alpha => Intervalo de confiança (1-alpha)
    alpha = .05

    # Média e desvio-padrão
    media = pre_max.mean().values.tolist()
    desvio= pre_max.std().values.tolist()

    # Parâmetros de locação e escala
    para_for = [None]*len(pre_max.columns)    
    if window.Element('distr').get()=='Gumbel':
        para_loc = [i-j*coefLin for i,j in zip(media,desvio)]
        para_esc = [i*coefAng for i in desvio]
    elif window.Element('distr').get()=='Lognormal' or window.Element('distr').get()=='Weibull':        
            #Para Weibull e Lognormal faz-se o cálculo dentro do loop abaixo
            para_loc = [None]*len(pre_max.columns)
            para_esc = [None]*len(pre_max.columns)
    
    # Probabilidade empírica
    prob = np.linspace(1/nes,1-1/nes,nes) 
   
    # Aceite do teste Kolmogorov-Smirnov
    kaceite=[['']*len(pre_max.columns)] 

    # Aceite do teste Chi-Square
    qaceite=[['']*len(pre_max.columns)] 

    # Status do Kolmogorov-Smirnov
    kstatus=[['']*len(pre_max.columns)] 

    # Status do Chi-Square
    qstatus=[['']*len(pre_max.columns)] 

    # Status do Kolmogorov-Smirnov p-value     
    kpvalue=[['']*len(pre_max.columns)] 

    # Status do Chi-Square     
    qpvalue=[['']*len(pre_max.columns)] 

    # Valor crítico do KS-Test (dependo do intevalo de confiança e do N amostral)    
    kcrival=[['{:,.4f}'.format(ks_crit_val(alpha, nes))]*len(pre_max.columns)] 
    qcrival=[['']*len(pre_max.columns)] 

    # Coeficiente de correlação
    coefcor=[['']*len(pre_max.columns)] 

    # Erro Médio Quadrático 
    rootmsq=[['']*len(pre_max.columns)] 
    
    # Teste Chi2 para aderência do ajuste.
    # Número de classe e Probabilidades para o Teste Chi2 para aderência do ajuste.
    k=max(5,np.floor(np.sqrt(nes))) # Regra da raiz quadrada para a definição do número de classes.
    k = min(k,20) # Hamdan(1963) Mais de 20 classes não aumentam significativamente a potência do teste. Pag.286  ESENWANGER O.M. Elementes of Statistical Analysis. General Climatology, 1B Elsevier., 1986
    dp = 1/k
    P = np.arange(dp,1+1e-6,dp)-1e-6 # Vetor com intervalos de mesma densidade de Probabilidades
       
    # Cálculo dos índices dos ajustes
    for duracao,ij in zip(pre_max.columns,range(len(pre_max.columns))):
        
        # Epocas de chuva para determinada duração
        chuva_epoca=pre_max[duracao][::-1]

        
        if window.Element('distr').get()=='Gumbel':
            # Bins do hsitograma de mesma densidade de probabilidade
            chu_bins = sc.stats.gumbel_r.isf(1-P,para_loc[ij],para_esc[ij])

            # Estimativa de chuva para para determinada duração
            chuva_ajust=sc.stats.gumbel_r.isf(1-prob, para_loc[ij], para_esc[ij])
            #chuva_p_cdf=sc.stats.gumbel_r.cdf(chuva_epoca, loc=para_loc[ij], scale=para_esc[ij])
            
            # Kolmogorov-Smirnov test
            kstatus[0][ij],kpvalue[0][ij] = sc.stats.kstest(chuva_epoca,sc.stats.gumbel_r.cdf,args=(para_loc[ij],para_esc[ij]))

        elif window.Element('distr').get()=='Weibull':        
            # ajuste dos parâmetros
            para_for[ij], para_loc[ij], para_esc[ij]  =  sc.stats.weibull_min.fit(pre_max.iloc[:,ij],loc=0,scale=1)
            para_loc[ij] = coefLin*para_loc[ij]
            para_esc[ij] = coefAng*para_esc[ij]
            para_for[ij] = coefFor*para_for[ij]

            # Bins do hsitograma de mesma densidade de probabilidade
            chu_bins = sc.stats.weibull_min.isf(1-P, para_for[ij], loc = para_loc[ij], scale = para_esc[ij])

            # Estimativa de chuva para para determinada duração
            chuva_ajust = sc.stats.weibull_min.isf(1-prob, para_for[ij], loc=para_loc[ij], scale=para_esc[ij])
            #chuva_p_cdf = sc.stats.weibull_min.cdf(chuva_epoca, para_for[ij], loc=para_loc[ij], scale=para_esc[ij])

            # Kolmogorov-Smirnov test
            kstatus[0][ij],kpvalue[0][ij] = sc.stats.kstest(chuva_epoca,sc.stats.weibull_min.cdf,args=(para_for[ij],para_loc[ij],para_esc[ij]))

        elif window.Element('distr').get()=='Lognormal':
            # ajuste dos parâmetros
            forma, locacao, escala = sc.stats.lognorm.fit(pre_max.iloc[:,ij])
            para_loc[ij]=coefLin*locacao
            para_esc[ij]=coefAng*escala
            para_for[ij]=coefFor*forma

            # Bins do hsitograma de mesma densidade de probabilidade
            chu_bins = sc.stats.lognorm.isf(1-P, para_for[ij], loc = para_loc[ij], scale = para_esc[ij])

            # Estimativa de chuva para para determinada duração
            chuva_ajust = sc.stats.lognorm.isf(1-prob, para_for[ij], loc = para_loc[ij], scale = para_esc[ij])                
            #chuva_p_cdf = sc.stats.lognorm.cdf(chuva_epoca, para_for[ij], loc=para_loc[ij], scale=para_esc[ij])

            # Kolmogorov-Smirnov test
            kstatus[0][ij],kpvalue[0][ij] = sc.stats.kstest(chuva_epoca,sc.stats.lognorm.cdf,args=(para_for[ij],para_loc[ij],para_esc[ij]))

        #.....................................................................
        # Kolmogorov-Smirnov test
        #.....................................................................
        #kstatus[0][ij],kpvalue[0][ij] = sc.stats.kstest(chuva_epoca,chuva_p_cdf)

        #.....................................................................
        # Chi-Square test
        #.....................................................................
        chu_binb = np.append(chu_bins.min()-1e6,np.append(chu_bins,chu_bins.max()))

        # histograma        
        nh,bedges = np.histogram(chuva_epoca,chu_binb)
        nh  = nh[nh!=0]  # retira as classes zeradas
        
        # Graus de liberdade        
        GL = len(nh)-1 

        qstatus[0][ij],qpvalue[0][ij] = sc.stats.chisquare(nh)
        
        # Chi-Square Critival Value
        qcrival[0][ij]=sc.stats.chi2.isf(alpha,GL)        
        
        # Correlação de Pearson e Erro médio quadrático
        coefcor[0][ij]=np.corrcoef(chuva_epoca,chuva_ajust )[0][1]**2
        rootmsq[0][ij]=np.sqrt(np.mean( (chuva_epoca - chuva_ajust )**2) )

        # Aceite da hipótese do teste de Kolmogorov-Smirnov
        if kpvalue[0][ij] >= alpha:
            kaceite[0][ij]='Sim'
        else:
            kaceite[0][ij]='Não'
            
        # Aceite da hipótese do teste Chi-quadrado
        if qpvalue[0][ij] >= alpha:
            qaceite[0][ij]='Sim'
        else:
            qaceite[0][ij]='Não'
            
    # Cabeçalho da tabela
    lhead = pre_max.columns.values.tolist()
    lhead.insert(0,' ')

    # Etiqueta das linhas da tabela Kolmogorov-Smirnov Test
    kaceite[0].insert(0,'Aceite')
    kstatus[0].insert(0,'Status')
    kcrival[0].insert(0,'Critical Value')
    kpvalue[0].insert(0,'p-value')
    coefcor[0].insert(0,'R(Corr.Coef^2)')
    rootmsq[0].insert(0,'RMS(mm)')
    
    # Etiqueta das linhas da tabela Chi-Square Test
    qaceite[0].insert(0,'Aceite')
    qstatus[0].insert(0,'Status')
    qcrival[0].insert(0,'Critical Value')
    qpvalue[0].insert(0,'p-value')
    #coefcor[0].insert(0,'R(Corr.Coef^2)')
    #rootmsq[0].insert(0,'RMS(mm)')
    
    # Acrescentando linhas a tabela Kolmogorov-Smirnov Test
    ktabela = kaceite + kstatus + kcrival + kpvalue  + coefcor + rootmsq

    # Acrescentando linhas a tabela Chi-Square Test
    qtabela = qaceite + qstatus + qcrival + qpvalue  + coefcor + rootmsq
   
    # Formatando os números da tabela para 04 dígitos decimais.
    #tabela = ktabela.copy()
    for ji in range(len(ktabela)):
        # Kolmogorov-Smirnov Test
        for kvalor,it in zip(ktabela[ji],range(len(ktabela[ji]))):
            if isinstance(kvalor,float):
                ktabela[ji][it] = '{:.4f}'.format(kvalor)
        # Chi-Square Test
        for qvalor,it in zip(qtabela[ji],range(len(qtabela[ji]))):
            if isinstance(qvalor,float):
                qtabela[ji][it] = '{:.4f}'.format(qvalor)

    return ktabela, qtabela, lhead, para_loc, para_esc, para_for

# =============================================================================
# # ----------- Tabela com os resultados dos testes de aderência ----------- 
# =============================================================================
def tab_aderencia(pre_max, coefLin, coefAng, nes,**kwargs):
    tipo = kwargs.get('tipo',[])
    ktabela, qtabela, lhead, _, _, _ = aderencia_2_lista(pre_max, coefLin, coefAng, nes,forma=float(window.Element('pforma').get()))    

    kaderencia=[]
    qaderencia=[]

    if (tipo=='kol') or (len(tipo)==0):
        klayout = [[
            sg.Table(values= ktabela,
                     headings=lhead,
                     font='Helvetica',
                     def_col_width=45,
                     col_widths = 100,
                     background_color='lightgrey',
                     alternating_row_color='white',
                     auto_size_columns=True,
                     key = 'wtab_kstest')
            ]]
        
        kaderencia = sg.Window('Aderência - Kolmogorov-Smirnov '+fn, klayout, grab_anywhere=False)
        kaderencia.read(timeout=1.0)
    
    elif (tipo=='chi') or (len(tipo)==0):
        qlayout = [[
            sg.Table(values= qtabela,
                     headings=lhead,
                     font='Helvetica',
                     def_col_width=45,
                     col_widths = 100,
                     background_color='lightgrey',
                     alternating_row_color='white',
                     auto_size_columns=True,
                     key = 'wtab_cstest')
            ]]
        
        qaderencia = sg.Window('Aderência - Chi-Square '+fn, qlayout, grab_anywhere=False)
        qaderencia.read(timeout=150)

    return kaderencia, qaderencia

# =============================================================================
# # --- Transforma o IDF em lista para apresentar como tabela no PySimleGui --- 
# =============================================================================
def estimativa_idf_2_lista(t_func, **kwargs):
    vmaxs = kwargs.get('vmaxs',[])
    
    t = np.array([int(x.replace('h','')) for x in t_func.columns ])

    # Conversão do tempo conforme a seleção do usuário
    if True in ['h' in x for x in t_func.columns.tolist()]:
        if window.Element('drop').get() != 'horas':
            t = t*60
    
    idf2precip = t_func*t
    idf2precip = idf2precip.applymap('{:,.4f}'.format)
    idf2precip =  idf2precip.append(pd.DataFrame([[' ']*idf2precip.shape[1]],index=[' '],columns=idf2precip.columns))
    if len(vmaxs):
        idf2precip = idf2precip.append(vmaxs)
    
    # Talvez o Andrioni otimize o bloco abaixo
    ldados = idf2precip.values.tolist()
    for x,y in zip(range(len(ldados)),idf2precip.index.values.tolist()):
        if isinstance(y,float) or isinstance(y,int):
            ldados[x].insert(0,str(int(y))+' anos (mm)')
        elif isinstance(y,str):
            ldados[x].insert(0,y)

    lhead = idf2precip.columns.values.tolist()
    lhead.insert(0,'Perído de Retorno')    

    return ldados, lhead

# =============================================================================
# # --- Tabela com as estimativas de precipitação a partir das curvas IDF --- 
# =============================================================================
def tab_estimativa_idf(t_func, **kwargs):
    vmaxs = kwargs.get('vmaxs',[])
    fn = kwargs.get('titulo',' ')    

    ldados, lhead = estimativa_idf_2_lista(t_func,vmaxs=vmaxs)

    clayout = [[
        sg.Table(values=ldados,
                 headings=lhead,
                 font='Helvetica',
                 def_col_width=45,
                 background_color='lightgrey',
                 alternating_row_color='white',
                 auto_size_columns=True,
                 pad=(25,25),
                 display_row_numbers=False,
                 key ='wtab_curv_idf'
                 )
        ]]

    wtab_curv_i = sg.Window(' Estimativas de Precipitação a partir das curvas IDF ' + fn, clayout, grab_anywhere=False)
    wtab_curv_i.read(timeout=150)
    
    return wtab_curv_i
    
# =============================================================================
# # ------------------------ Tabela com as erros e RMS -----------------------
# =============================================================================
def diferenca_rms_2_lista(t_func, xt):

    t = np.array([float(x.replace('h','')) for x in t_func.columns ])
    
    # Conversão do tempo conforme a seleção do usuário
    if True in ['h' in x for x in t_func.columns.tolist()]:
        if window.Element('drop').get() != 'horas':
            t = t*60

    # Precipitação a partir das curvas IDFs
    idf2precip = t_func*t
    
    # Diferença entre a Precipitação das curvas IDF e a estimada por Gumbel
    dif_estim_pre = idf2precip-xt
    
    # RMS
    emq = dif_estim_pre.pow(2).mean().pow(1/2)
    #emq = emq.astype(float)
    #emq = emq.map('{:.4f}'.format)
    
    # Adiciona linha vazia na matriz
    #dif_estim_pre =  dif_estim_pre.append(pd.DataFrame([[' ']*dif_estim_pre.shape[1]],index=[' '],columns=dif_estim_pre.columns))
    
    # Adiciona linha do Erro médio quadrático
    dif_estim_pre =  dif_estim_pre.append(pd.DataFrame([emq],index=['RMS (mm)'],columns=emq.index))
    dif_estim_pre =  dif_estim_pre.append(pd.DataFrame([emq],index=['RMS (mm)'],columns=emq.index))
    
    dif_estim_pre = dif_estim_pre.applymap('{:.4f}'.format)
    dif_estim_pre = dif_estim_pre.astype(float)

    linhas = dif_estim_pre.index.tolist()    
    linhas[-2] = ' '
       
    dif_estim_pre.index = linhas
    dif_estim_pre.loc[' ',:]=' ' 
    
    # Talvez o Andrioni otimize o bloco abaixo
    ldados = dif_estim_pre.values.tolist()
    for x,y in zip(range(len(ldados)),dif_estim_pre.index.values.tolist()):
        if isinstance(y,float) or isinstance(y,int):
            ldados[x].insert(0,str(int(y))+' anos (mm)')
        elif isinstance(y,str):
            ldados[x].insert(0,y)
                  
    lhead = dif_estim_pre.columns.values.tolist()
    lhead.insert(0,'Perído de Retorno')    

    return ldados, lhead 
    
# =============================================================================
# # ----------------------- Tabela com as erros e RMS  ------------------------ 
# =============================================================================
def tab_diferenca_rms(t_func, xt, **kwargs):
    fn = kwargs.get('titulo',' ')

    ldados, lhead = diferenca_rms_2_lista(t_func, xt)

    dlayout = [[
        sg.Table(values=ldados,
                 headings=lhead,
                 font='Helvetica',
                 def_col_width=45,
                 background_color='lightgrey',
                 alternating_row_color='white',
                 auto_size_columns=True,
                 pad=(25,25),
                 display_row_numbers=False,
                 key ='wtab_dif_rms'
                 )
        ]]

    wtab_dif_e_rms = sg.Window(' Diferença & RMS ( IDF curves - Precipitation(Gumbel) ) ' + fn, dlayout, grab_anywhere=False)
    wtab_dif_e_rms.read(timeout=150)
    
    return wtab_dif_e_rms

# =============================================================================
# # ----------------------- Atualização das Tabelas
def atualiza_tabelas(values,vlocais):
    
    #........................................................
    # Tabela de Gumbel
    #........................................................
    # Decisão de abertura da tabela das estimativas de de Extremos
    if values['estimativas_extremos']:
        xt = window.Element('estimativas_extremos').metadata
        vmaxs = window.Element('tempo_total').metadata['valores_maximos']
        fn = window.Element('-loaded-').get()
        
        #..........................................
        # Tabela existe
        if 'wtab_estimativa_extremos' in vlocais:
                wtab_estimativa_extremos = vlocais.get('wtab_estimativa_extremos')
                # Tabela foi fechada
                if wtab_estimativa_extremos.was_closed():
                    wtab_estimativa_extremos = tab_estimativa_extremos(xt,vmaxs = vmaxs ,titulo = fn)

                # Tabela está aberta
                elif not wtab_estimativa_extremos.was_closed():
                    wtab_estimativa_extremos.set_title(wtab_estimativa_extremos.Title[0:wtab_estimativa_extremos.Title.find(':')-1]+ window.Element('-loaded-').get())
                    ldados,lhead = df_ext_2_lista(xt,vmaxs=vmaxs)
                    wtab_estimativa_extremos.Element('wtab_gumbel').update(values=ldados)
                    wtab_estimativa_extremos.Element('wtab_gumbel').ColumnHeadings=lhead

        #..........................................
        # Tabela não existe
        elif 'wtab_estimativa_extremos' not in vlocais:
            vmaxs = window.Element('tempo_total').metadata['valores_maximos']
            wtab_estimativa_extremos = tab_estimativa_extremos(xt,vmaxs = vmaxs ,titulo = fn)

    # Decisão de fechamento da tabela das estimativas de Extremos
    elif values['estimativas_extremos']==False:                
        if 'wtab_estimativa_extremos' in vlocais:
            wtab_estimativa_extremos = vlocais.get('wtab_estimativa_extremos')
            wtab_gumbel_element = wtab_estimativa_extremos.Element('wtab_gumbel', silent_on_error=True)
            if wtab_gumbel_element:
                wtab_estimativa_extremos.close()
        elif not 'wtab_estimativa_extremos' in vlocais:
            wtab_estimativa_extremos = []

    #........................................................
    # Tabelas de aderência
    #........................................................
    # Decisão de abertura da tabela de testes aderência

    if values['aderencia_check']:
        pre_max = window.Element('epocas').metadata
        coefLin = float(window.Element('linear').get())
        coefAng = float(window.Element('angular').get()) # coeficiente angular #
        nes = int(window.Element('epocas').get())

        #..........................................
        # Tabela do Kolmogorov-Smirnov test existe
        if 'wtab_kaderencia' in vlocais:
            wtab_kaderencia = vlocais.get('wtab_kaderencia')
            # Tabela foi fechada
            if wtab_kaderencia.was_closed():
                wtab_kaderencia, _ = tab_aderencia(pre_max, coefLin, coefAng, nes, tipo='kol')

            # Tabela está aberta
            elif not wtab_kaderencia.was_closed():
                wtab_kaderencia.set_title(wtab_kaderencia.Title[0:wtab_kaderencia.Title.find(':')-1]+ window.Element('-loaded-').get())
                kldados, _, lhead, _, _, _ = aderencia_2_lista(pre_max, coefLin, coefAng, nes,forma=float(window.Element('pforma').get()))
                wtab_kaderencia.Element('wtab_kstest').update(values=kldados)
                wtab_kaderencia.Element('wtab_kstest').ColumnHeadings=lhead
                
        #..........................................
        # Tabela do Kolmogorov-Smirnov test não existe
        elif not 'wtab_kaderencia' in vlocais:
            wtab_kaderencia, _ = tab_aderencia(pre_max, coefLin, coefAng, nes, tipo='kol')
        
        #..........................................
        # Tabela do Chi-test existe
        if 'wtab_qaderencia' in vlocais:
            wtab_qaderencia = vlocais.get('wtab_qaderencia')
            # Tabela foi fechada
            if wtab_qaderencia.was_closed():
                _, wtab_qaderencia = tab_aderencia(pre_max, coefLin, coefAng, nes, tipo='chi')

            # Tabela está aberta
            elif not wtab_qaderencia.was_closed():
                wtab_qaderencia.set_title(wtab_qaderencia.Title[0:wtab_qaderencia.Title.find(':')-1]+ window.Element('-loaded-').get())
                _, qldados, lhead, _, _, _ = aderencia_2_lista(pre_max, coefLin, coefAng, nes, forma=float(window.Element('pforma').get()))
                wtab_qaderencia.Element('wtab_cstest').update(values=qldados)
                wtab_qaderencia.Element('wtab_cstest').ColumnHeadings=lhead

        #..........................................
        # Tabela do Chi-test não existe
        elif 'wtab_qaderencia' not in vlocais:
            _, wtab_qaderencia = tab_aderencia(pre_max, coefLin, coefAng, nes, tipo='chi')
            
    # Decisão de fechamento das tabelas de aderência
    elif values['aderencia_check']==False:
        if 'wtab_kaderencia' in vlocais:
            wtab_kaderencia = vlocais.get('wtab_kaderencia')
            wtab_kaderencia_element = wtab_kaderencia.Element('wtab_kstest', silent_on_error=True)
            if wtab_kaderencia_element:
                wtab_kaderencia.close()
        elif not 'wtab_kaderencia' in vlocais:
            wtab_kaderencia=[]

        if 'wtab_qaderencia' in vlocais:
            wtab_qaderencia = vlocais.get('wtab_qaderencia')
            wtab_qaderencia_element = wtab_qaderencia.Element('wtab_cstest', silent_on_error=True)
            if wtab_qaderencia_element:
                wtab_qaderencia.close()
        elif not 'wtab_qaderencia' in vlocais:
            wtab_qaderencia=[]

    #........................................................
    # Tabelas de precipitação oriunda das curvas IDF
    #........................................................
    # Decisão de abertura da tabela das estimativas a partir das curvas IDF
    if values['idf']:
        t_func = window.Element('idf').metadata
        vmaxs = window.Element('tempo_total').metadata['valores_maximos']
        fn = window.Element('-loaded-').get()

        #..........................................
        # Tabela das estimativas de precip. IDF existe
        if 'wtab_curv_p' in vlocais:
            wtab_curv_p = vlocais.get('wtab_curv_p')

            # Tabela foi fechada
            if wtab_curv_p.was_closed():
                wtab_curv_p = tab_estimativa_idf(t_func,titulo = fn,vmaxs = vmaxs)                                

            # Tabela está aberta
            elif not wtab_curv_p.was_closed():
                wtab_curv_p.set_title(wtab_curv_p.Title[0:wtab_curv_p.Title.find(':')-1]+fn)
                ldados, lhead = estimativa_idf_2_lista(t_func,vmaxs=vmaxs)
                wtab_curv_p.Element('wtab_curv_idf').update(values=ldados)
                wtab_curv_p.Element('wtab_curv_idf').ColumnHeadings=lhead
                
        #..........................................
        # Tabela das estimativas de precip. IDF não existe
        elif not 'wtab_curv_p' in vlocais:
            wtab_curv_p = tab_estimativa_idf(t_func,titulo = fn,vmaxs = vmaxs)            
    
    # Decisão de fechamento das tabelas de aderência                
    elif values['idf']==False:
        if 'wtab_curv_p' in vlocais:
            wtab_curv_p = vlocais.get('wtab_curv_p')
            wtab_curv_p_element = wtab_curv_p.Element('wtab_curv_idf', silent_on_error=True)
            if wtab_curv_p_element:
                wtab_curv_p.close()
        elif not'wtab_curv_p' in vlocais:
            wtab_curv_p=[]

    #........................................................
    # Tabelas de Erro e RMS
    #........................................................
    # Decisão de abertura da tabela de erro e RMS entra as estimativas de precipitação de Gumbel e IDF
    if values['RMS']:
        t_func = window.Element('idf').metadata
        xt = window.Element('estimativas_extremos').metadata            
        fn = window.Element('-loaded-').get()
        
        #..........................................
        # Tabela das estimativas de erro e RMS existe
        if 'wtab_dif_e_rms' in vlocais:
            wtab_dif_e_rms = vlocais.get('wtab_dif_e_rms')
            
            # Tabela foi fechada
            if wtab_dif_e_rms.was_closed():
                wtab_dif_e_rms = tab_diferenca_rms(t_func, xt, titulo = fn)

            # Tabela está aberta
            elif not wtab_dif_e_rms.was_closed():
                wtab_dif_e_rms.set_title(wtab_dif_e_rms.Title[0:wtab_dif_e_rms.Title.find(':')-1] + fn)
                ldados, lhead = diferenca_rms_2_lista(t_func, xt)
                wtab_dif_e_rms.Element('wtab_dif_rms').update(values=ldados)
                wtab_dif_e_rms.Element('wtab_dif_rms').ColumnHeadings=lhead
        
        #..........................................
        # Tabela das estimativas de erro e RMS não existe
        elif  not 'wtab_dif_e_rms' in vlocais:
            wtab_dif_e_rms = tab_diferenca_rms(t_func, xt, titulo = fn)
            
    # Decisão de fechamento das tabelas de erro e RMS
    elif values['RMS']==False:
        if 'wtab_dif_e_rms' in vlocais:
            wtab_dif_e_rms = vlocais.get('wtab_dif_e_rms')
            wtab_dif_e_rms_element = wtab_dif_e_rms.Element('wtab_dif_rms')
            if wtab_dif_e_rms_element:
                wtab_dif_e_rms.close()
        elif not 'wtab_dif_e_rms' in vlocais:
            wtab_dif_e_rms=[]
    
    return wtab_estimativa_extremos, wtab_kaderencia, wtab_qaderencia, wtab_curv_p, wtab_dif_e_rms

# =============================================================================
# # --------------------- plotagem da Seleção de Máimos --------------------- 
# =============================================================================
def plota_sel_max(window,acao):
    
    if not acao:
        if window['max_ind'].metadata: # Desmarca apaga as figuras (não consegue apagar as figuras do painel lateral)
            ax_selmax = window['max_ind'].metadata
            for  j in range(len(ax_selmax)):
                plt.close(ax_selmax[j][0].figure)
            window['max_ind'].metadata = None
    else: # Marca plota as figuras (No Spyder plota no painel lateral)
        ax_selmax=[]
        for icol in range(0,len(df.columns),3):
            fig, axes = plt.subplots(nrows=3, ncols=1)
            # Plota a série completa
            df.iloc[:,icol:icol+3].plot(subplots=True,kind='line',grid=True,linestyle ='--',linewidth=1,ax=axes)
            
            # Plota a seleção de épocas (máximos)
            prec_max_plot.iloc[:,icol:icol+3].plot(subplots=True,kind='line',title = 'Seleção de Épocas',
                      ylabel='mm',marker='d',ax=axes)
            ax_selmax.append(axes)

        plt.show()
        window['max_ind'].metadata = ax_selmax

# Geração do Log de saída da análise de curvas IDF extremas
def gera_log(window):
    
    # Épocas
    pre_max = window.Element('epocas').metadata
    
    # Estimativas de Gumbel
    xt = window.Element('estimativas_extremos').metadata
    lldados,llhead = df_ext_2_lista(xt)
    #ext_g = [str(lldados[i])[2:-2].replace(',','').replace("'",'') for i in range(len(lldados))]
    ext_g = [None]*len(lldados)
    for ii in range(len(lldados)):
        ext_g[ii] = str([ aa.rjust(14,' ') if isinstance(aa,str) else '{:8.3f}'.format(aa) for aa in lldados[ii]])[2:-2].replace(',','').replace("'",'')


    # Testes de Aderência
    kldados,qldados,llhead,para_loc,para_esc,para_for = aderencia_2_lista(pre_max, coefLin, coefAng, nes,forma=float(window.Element('pforma').get()))
    #kst_a = [str(kldados[i])[2:-2].replace(',','').replace("'",'') for i in range(len(kldados))]
    #qst_a = [str(qldados[i])[2:-2].replace(',','').replace("'",'') for i in range(len(qldados))]
    kst_a = [None]*len(kldados)
    qst_a = [None]*len(qldados)
    for ii in range(len(kldados)):
        kst_a[ii] = str([ aa.rjust(14,' ') if isinstance(aa,str) else '{:8.3f}'.format(aa) for aa in kldados[ii]])[2:-2].replace(',','').replace("'",'')
        qst_a[ii] = str([ aa.rjust(14,' ') if isinstance(aa,str) else '{:8.3f}'.format(aa) for aa in qldados[ii]])[2:-2].replace(',','').replace("'",'')

    # Estimativas de precipitação a patir das IDFs
    t_func = window.Element('idf').metadata
    lldados,llhead = estimativa_idf_2_lista(t_func)
    #idf_p = [str(lldados[i])[2:-2].replace(',','').replace("'",'') for i in range(len(lldados))]
    idf_p = [None]*len(lldados)
    for ii in range(len(lldados)):
        idf_p [ii] = str([ aa.rjust(14,' ') if isinstance(aa,str) else '{:8.3f}'.format(aa) for aa in lldados[ii]])[2:-2].replace(',','').replace("'",'')


    # Diferença e RMS
    lldados,llhead = diferenca_rms_2_lista(t_func, xt)
    #dif_r = [str(lldados[i])[2:-2].replace(',','').replace("'",'') for i in range(len(lldados))]
    dif_r = [None]*len(lldados)
    for ii in range(len(lldados)):
        dif_r [ii] = str([ aa.rjust(14,' ') if isinstance(aa,str) else '{:8.3f}'.format(aa) for aa in lldados[ii]])[2:-2].replace(',','').replace("'",'')
    
    distribuicao = {
        'Gumbel':' F(Precip)= exp(-exp(-escala*(Precip-loc)))',
        'Lognormal':'F(Precip) = 1/(forma * Precip_n * sqrt(2*pi)) * exp(-log^2(Precip_n)/(2*forma^2))  onde Precip_n = ((Precip-loc)/escala)',
        'Weibull':' F(Precip)= 1-exp(-[(Precip-loc)/escala]^forma)'
                    }
    if window.Element('distr').get()=='Weibull' or window.Element('distr').get()=='Lognormal':
        tpara_for =[ ' forma   = ' + str(['{:8.2f}'.format(para) for para in para_for])[1:-1].replace(",",'').replace("'",'')]
        relacao = [' Proporção dos parâmetros',
                   '====================================================================',
                   '  Escala    Locacao    Forma',
                   '  ' + window.Element('angular').get() + '    ' + window.Element('linear').get() + '     ' + window.Element('pforma').get(),
                   ' ',
                   '  que multiplicam as estimativas dos parâmetros da distribuição de ' + window.Element('distr').get()]
        
    elif window.Element('distr').get()== 'Gumbel':
        tpara_for = []
        relacao = [' Relação de Assimetria Gumbel-Gauss',
                   '====================================================================',
                   '  Y = -ln ( -ln ( 1 -1/(T*nrpa) ) )',
                   '  Precip(T) =   media  + (' + window.Element('angular').get() + ' * Y + ' + window.Element('linear').get() + ') * desvio-padrão',
                   ' ',
                   ' onde Y = variável reduzida de Gumbel, T = período de retorno em anos e nrpa = número de registros por ano']

    if window.Element('tipo_idf').get() == 'exponencial':
        eq_curva_idf = '  i(mm/h) = (' + '{:.4f}'.format(float(window.Element('K_ou_m_log_natural').get())) + '*T.^' + '{:.4f}'.format(float(window.Element('M').get())) + ').*((t+' + '{:.4f}'.format(float(window.Element('t0').get())) + ').^' + '{:.4f}'.format(-float(window.Element('n').get())) +')'
    elif window.Element('tipo_idf').get() == 'logarítmica natural':
        eq_curva_idf = '  i(mm/h) = (' + '{:.4f}'.format(float(window.Element('K_ou_m_log_natural').get())) + '* ln(T) ' + '{:+.4f}'.format(float(window.Element('K_lognatural').get())) + ').*((t ' + '{:+.4f}'.format(float(window.Element('t0').get())) + ').^' + '{:.4f}'.format(-float(window.Element('n').get())) +')'
        
    logtxt = ['Titulo da rodada do programa curva_idf : ' + window.Element('nome').get(),
    'Data de execucao do programa: ' + datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
    'Arquivo de entrada dos dados de precipitação: ' + window.Element('-loaded-').get(),
    'Numero registros da série temporal: ' + window.Element('n_amostral').get(),
    'Quantidades de durações de chuvas: ' + str(len(window.Element('estimativas_extremos').metadata.columns)),
    'Extensão temporal da serie: ' + window.Element('tempo_total').get() + ' ' + window.Element('combo').get(),
    ' ',
    ' ',
    '====================================================================',
    ' Seleção de Máximos',
    '====================================================================',
    '** Intervalo de tempo em dias para a seleção de épocas: ' + window.Element('sel_max').get(),
    ' ',
    '** Número de registros por ano (~365/DT -  NRPA): ' + window.Element('por_ano').get(),
    ' ',
    '** Numero amostral de máxs. selecionados: ' + window.Element('epocas').get(),
    ' ',
    '** Limiar inferior de seleção de precip.(mm): ' + window.Element('limiar').get(),
    ' ',
    ' ',
    '====================================================================',
    ' Estatística básica das epocas',
    '====================================================================',
    ' duração             ' + str([aa.rjust(8,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",''),
    ' media         =     ' + str(['{:8.2f}'.format(mmm) for mmm in pre_max.mean().tolist()])[1:-1].replace(",",'').replace("'",''),
    ' desvio-padrão =     ' + str(['{:8.2f}'.format(mmm) for mmm in pre_max.std().tolist()])[1:-1].replace(",",'').replace("'",''),
    ' máximos       =     ' + str(['{:8.2f}'.format(mmm) for mmm in pre_max.max().tolist()])[1:-1].replace(",",'').replace("'",''),
    ' ',
    ' ',
    '===================================================================='] + \
    relacao + \
    [' ',
    ' ',
    '====================================================================',
    ' Estatística de Extremos com ajuste de '+ window.Element('distr').get() +' - Precipitação em mm',
    '===================================================================='] + \
    [distribuicao[window.Element('distr').get()]] + \
    [' ',
    ' duração  ' + str([aa.rjust(8,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",''),
    ' locação = ' + str(['{:8.2f}'.format(para) for para in para_loc])[1:-1].replace(",",'').replace("'",''),
    ' escala  = ' + str(['{:8.2f}'.format(para) for para in para_esc])[1:-1].replace(",",'').replace("'",'')] + \
    tpara_for + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Estimativas através da distribuição de '+ window.Element('distr').get() +' - Precipitação em mm',
    '---------------------------------------------------------------------------------',
    ' duração     ' + str([aa.rjust(14,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",'')] + \
    ext_g + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Aderência Kolmogorov-Smirnov',
    '---------------------------------------------------------------------------------',
    ' duração     ' + str([aa.rjust(14,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",'')] + \
    kst_a + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Aderência Chi-Square',
    '---------------------------------------------------------------------------------',
    ' duração     ' + str([aa.rjust(14,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",'')] + \
    qst_a + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Estimativas a partir das Curvas IDF (mm)',
    '---------------------------------------------------------------------------------',
    ' duração     ' + str([aa.rjust(14,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",'')] + \
    idf_p + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Diferencas entre as estimativas IDF e '+ window.Element('distr').get() +' - Precipitação em mm',
    '---------------------------------------------------------------------------------',
    ' duração       ' + str([aa.rjust(8,' ') for aa in window.Element('idf').metadata.columns.tolist()])[2:-2].replace(',','').replace("'",'')] + \
    dif_r  + \
    [' ',
    ' ',
    '---------------------------------------------------------------------------------',
    ' Equação de chuvas intensas  - Intensidade em mm/h',
    '---------------------------------------------------------------------------------',
    eq_curva_idf,
    #'  i(mm/h) = (' + '{:.4f}'.format(float(window.Element('K_ou_m_log_natural').get())) + '*T.^' + '{:.4f}'.format(float(window.Element('M').get())) + ').*((t+' + '{:.4f}'.format(float(window.Element('t0').get())) + ').^' + '{:.4f}'.format(-float(window.Element('n').get())) +')',
    '  onde T = período de retorno em anos e t = duração em minutos',
    ' ',
    ' ']

    return logtxt        


# =============================================================================
# # ------ Loop & Process button menu choices ------ #      
# =============================================================================
read_successful = False
max_ind = False
gumbel = False
aderencia_check = False
idf = False
RMS = False

while True:      
    event, values = window.read()
    loaded_text = window['-loaded-']
    if event == sg.WIN_CLOSED or event == 'Cancelar':
        break     
    # ------ Process menu choices ------ #      
    if event == '-READ-':
        if values['max_ind']==True:
            max_ind=True
        if values['estimativas_extremos']==True:
            gumbel=True
        if values['aderencia_check']==True:
            aderencia_check=True
        if values['idf']==True:
            idf=True
        if values['RMS']==True:
            RMS=True

        # Leitura da chuva e algumas informações
        
        df, header_list, fn, tempo_decorr, ia = read_table()
        # Obtém informações de entrada dos objetos da interface gráfica
        DT = np.round(tempo_decorr) # na primeira estimativa é o tempo de decorrelação
        
        # limiar de seleção de épocas
        lim_inf = float(values['limiar'])
                
        # seleção de épocas para as diferentes durações de chuva
        # (rocesso mais demorado!!! Otimizações são bem-vindas
        pre_max, prec_max_plot = selecao_max_ind(df.copy(),DT,limiar=[lim_inf, np.nan])

        # Coeficientes do ajuste linear de Gumbel
        coefLin = float(values['linear'])
        coefAng = float(values['angular'])
        p_forma = float(values['pforma'])
        
        # Períodos de retorno
        T =[float(x) for x in window.Element('period_retorn').get().split()]# periodo de retorno em anos
        
        # Calcula as estimativas de precipitação
        if window.Element('distr').get()=='Gumbel':
            xt, nes, nrpa = calc_values_precip_gumbel(pre_max,T,coefLin,coefAng)
        
        elif window.Element('distr').get()=='Weibull':
            xt, nes, nrpa = calc_values_precip_weibull(pre_max,T,coefLin,coefAng,forma=p_forma)
        
        elif window.Element('distr').get()=='Lognormal':
            xt, nes, nrpa = calc_values_precip_lnorma(pre_max,T,coefLin,coefAng,forma=p_forma)

        maximo = pre_max

        # Calcula as estimativas das curvas IDF
        t_func, intensid, diference, K, m, t0, n = calc_curvas(xt)

        # insere o nome do arquivo na caixa de texto
        loaded_text.update("{}".format(fn))

        # Atualiza as figuras
        delete_figure_agg(figure_agg)
        delete_figure_agg2(figure_agg2)
        fig = plot_fig(pre_max,xt)
        fig2 = plot_fig2(t_func,xt)
        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)  # draw the figure
        figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)  # draw the figure
   
    if event == 'combo':
        tempo_total = window.Element('tempo_total').metadata['tempo_total']
        if values['combo'] == 'anos':
            window.Element('tempo_total').update('{:.3f}'.format(tempo_total))
        elif values['combo'] == 'meses':
            window.Element('tempo_total').update('{:.3f}'.format(tempo_total*12)) # tempo_total #
        elif values['combo'] == 'dias':
            window.Element('tempo_total').update('{:.3f}'.format(tempo_total*12*30.4375)) # tempo_total #
    
    # ------ Process menu choices ------ #      
    if event == 'max_ind':
        plota_sel_max(window,values['max_ind'])    
        
    if (event == 'estimativas_extremos' and  event == 'aderencia_check' and event == 'idf' and event == 'RMS'):
        wtab_estimativa_extremos, wtab_kaderencia, wtab_qaderencia, wtab_curv_p, wtab_dif_e_rms = atualiza_tabelas(values,locals())

    elif (event == 'estimativas_extremos' and  event == 'aderencia_check' and event == 'idf' ):
        wtab_estimativa_extremos, wtab_kaderencia, wtab_qaderencia, wtab_curv_p, _ = atualiza_tabelas(values,locals())
    
    elif (event == 'estimativas_extremos' and  event == 'aderencia_check'):
        wtab_estimativa_extremos, wtab_kaderencia, wtab_qaderencia, _, _ = atualiza_tabelas(values,locals())
    
    elif (event == 'estimativas_extremos' and  event == 'idf'):
        wtab_estimativa_extremos, _, _, wtab_curv_p, _ = atualiza_tabelas(values,locals())

    elif (event == 'estimativas_extremos' and  event == 'RMS'):
        wtab_estimativa_extremos, _, _, _, wtab_dif_e_rms = atualiza_tabelas(values,locals())

    elif (event == 'estimativas_extremos'):
        wtab_estimativa_extremos, _, _, _, _ = atualiza_tabelas(values,locals())
    
    elif (event == 'aderencia_check' and event == 'idf' and event == 'RMS'):
        _, wtab_kaderencia, wtab_qaderencia, wtab_curv_p, wtab_dif_e_rms = atualiza_tabelas(values,locals())
    
    elif (event == 'idf' and event == 'RMS'):
        _, _, wtab_qaderencia, wtab_curv_p, wtab_dif_e_rms = atualiza_tabelas(values,locals())
    
    elif (event == 'idf'):
        _, _, _, wtab_curv_p, _ = atualiza_tabelas(values,locals())

    elif ( event == 'aderencia_check' and event == 'idf'):
        _, wtab_kaderencia, wtab_qaderencia, wtab_curv_p, _ = atualiza_tabelas(values,locals())
    
    elif (event =='aderencia_check' and event == 'RMS'):
        _, wtab_kaderencia, wtab_qaderencia, _, wtab_dif_e_rms = atualiza_tabelas(values,locals())

    elif (event == 'aderencia_check'):
        _, wtab_kaderencia, wtab_qaderencia, _, _ = atualiza_tabelas(values,locals())

    elif (event == 'RMS'):
        _, _, _, _, wtab_dif_e_rms = atualiza_tabelas(values,locals())
        
    # Escoha da distribuição de extemos
    #if event == 'distr':
    
        # Ajuste manual de hora ou minuto para cáculo das curvas IDF
    if event == 'drop':

        # recalcula as curvas IDFs com base na escolha da unidade de tempo (hora ou minuto)
        calc_curvas(xt)

        # Atuliza o plot 2
        t_func = window.Element('idf').metadata
        
        delete_figure_agg2(figure_agg2)
        fig2 = plot_fig2(t_func,xt)
        figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)  # draw the figure
        
        # Atualiza as tabelas
        atualiza_tabelas(values,locals())

    # Ajuste manual de t0
    #if event == 't0':
    if (event == 'botao_t0' or event == 'tipo_idf'):

        xt = window.Element('estimativas_extremos').metadata
        t0 = float(window.Element('t0').get())
        
        # Calcula as estimativas das curvas IDF
        t_func, intensid, diference, K, m, t0, n = calc_curvas(xt,t0=t0)

        # Atualiza as figuras
        delete_figure_agg2(figure_agg2)
        fig2 = plot_fig2(t_func,xt)
        figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)  # draw the figure
        
        # Atualiza as tabelas
        atualiza_tabelas(values,locals())

    # Ajuste manual dos coeficientes da Relação de Assimetria Gumbel/Gauss
    #if (event=='angular' or event=='linear' or event=='pforma' or event=='period_retorn' or event=='distr'):
    if (event=='ajustar' or event=='period_retorn' or event=='distr'):

        if values['distr']=='Gumbel':
            
            # texto do Painel
            window.Element('painel_distr').update('Relação de Assimetria de Gumbel-Gauss')
            
            # valor dos spins
            #window.Element('angular').update('0.7797')
            #window.Element('linear').update('-0.450')
            window.Element('pforma').update(visible=False)
            
            # textos dos parâmetros
            window.Element('t_pforma').update(visible=False)
            window.Element('t_angular').update('Coef. Angular')
            window.Element('t_linear').update('Coef. Linear')
            
        else:

            # texto do Painel
            window.Element('painel_distr').update('Proporção dos parâmetros')

            # valor dos spins
            #window.Element('angular').update('1.0000')
            #window.Element('linear').update('1.000')

            # textos dos parâmetros
            window.Element('t_angular').update('Escala')
            window.Element('t_linear').update('Locação')

            # visibilidade do spin
            window.Element('pforma').update(visible=True)
            
            # visibilidade do texto do spin
            window.Element('t_pforma').update(visible=True)
        
        # Epocas
        pre_max = window.Element('epocas').metadata
        
        # Coeficientes do ajuste linear de Gumbel
        coefLin = float(values['linear'])
        coefAng = float(values['angular'])
        p_forma = float(values['pforma'])
        
        # Períodos de retorno
        T =[float(x) for x in window.Element('period_retorn').get().split()]# periodo de retorno em anos
        
        # Calcula as estimativas de precipitação
        if window.Element('distr').get()=='Gumbel':
            xt, nes, nrpa = calc_values_precip_gumbel(pre_max,T,coefLin,coefAng)
        
        elif window.Element('distr').get()=='Weibull':
            xt, nes, nrpa = calc_values_precip_weibull(pre_max,T,coefLin,coefAng,forma=p_forma)
        
        elif window.Element('distr').get()=='Lognormal':
            xt, nes, nrpa = calc_values_precip_lnorma(pre_max,T,coefLin,coefAng,forma=p_forma)
            
            
        # Calcula as estimativas das curvas IDF
        t_func, intensid, diference, K, m, t0, n = calc_curvas(xt)

        # Atualiza as figuras
        delete_figure_agg(figure_agg)
        delete_figure_agg2(figure_agg2)
        fig = plot_fig(pre_max,xt)
        fig2 = plot_fig2(t_func,xt)
        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)  # draw the figure
        figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)  # draw the figure

        # Atualiza as tabelas
        atualiza_tabelas(values,locals())
    
    # Ajuste manual da seleção de máximos
    if event == 'retry':
    #if (event == 'sel_max' or event == 'limiar'):
        
        # DataFrame original        
        df = window.Element('-loaded-').metadata

        # Obtém informações de entrada dos objetos da interface gráfica
        DT = float(window.Element('sel_max').get()) # na primeira estimativa é o tempo de decorrelação
        
        # limiar de seleção de épocas
        lim_inf = float(window.Element('limiar').get())

        # seleção de épocas para as diferentes durações de chuva
        pre_max, prec_max_plot = selecao_max_ind(df.copy(),DT,limiar=[lim_inf, np.nan])
        
        # Calculo do percentual
        percentual = sc.stats.percentileofscore(pre_max.stack().reset_index()[0],lim_inf, kind='rank')
        # window.Element('percentual').update('{:.0f}'.format(percentual)+"%")
        
        # Coeficientes do ajuste linear de Gumbel
        coefLin = float(values['linear'])
        coefAng = float(values['angular'])
        p_forma = float(values['pforma'])
        
        # Períodos de retorno
        T =[float(x) for x in window.Element('period_retorn').get().split()]# periodo de retorno em anos
        
        # Calcula as estimativas de precipitação
        if window.Element('distr').get()=='Gumbel':
            xt, nes, nrpa = calc_values_precip_gumbel(pre_max,T,coefLin,coefAng)
        
        elif window.Element('distr').get()=='Weibull':
            xt, nes, nrpa = calc_values_precip_weibull(pre_max,T,coefLin,coefAng,forma=p_forma)
        
        elif window.Element('distr').get()=='Lognormal':
            xt, nes, nrpa = calc_values_precip_lnorma(pre_max,T,coefLin,coefAng,forma=p_forma)

        
        # Calcula as estimativas das curvas IDF
        t_func, intensid, diference, K, m, t0, n = calc_curvas(xt)

        # Atualiza as figuras
        delete_figure_agg(figure_agg)
        delete_figure_agg2(figure_agg2)
        fig = plot_fig(pre_max,xt)
        fig2 = plot_fig2(t_func,xt)
        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)  # draw the figure
        figure_agg2 = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)  # draw the figure

        # Atualiza os gráficos de seleção de máximos
        plota_sel_max(window,values['max_ind'])    
        
        # Atualiza as tabelas
        atualiza_tabelas(values,locals())
    if event == 'salvamento':
        log = gera_log(window)
        with open(values['salvamento'], 'w') as f:
            f.write('\n'.join(log))
            
        sg.Popup(values['salvamento'], keep_on_top=True, title='Arquivo salvo com sucesso')

window.close()
