import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import prince
import string
import pickle

###################
#Funciones
def listAlphabet():
      return list(string.ascii_uppercase)

def MCA_score(indices,k_vecinos,time_index):
      lv_val = [-0.001, 0.25, 0.5, 1.0001]
      lv_name = ['Bajo','Medio','Alto']
      cl_ax = ['red','orange','green']
      s = 1/(np.exp(1)*((k_vecinos+10)**(1/3)))
      bd_mca = pd.concat([pd.cut(H_matrix.iloc[k_vecinos,:].reset_index(drop=True), labels=['A.P','E','P'],bins=[0, 0.5-s, 0.5+s, 1])]+
                [pd.cut(indices.reset_index(drop=True).iloc[:,ts],lv_val,labels= lv_name) for ts in range(indices.shape[1])], axis=1)
      bd_mca = bd_mca.astype(object)
      bd_mca.index = time_index
      names = listAlphabet()[:indices.shape[1]]
      bd_mca.columns =  ['Hurst']+list(listAlphabet()[:len(indices.columns)])
      Z = pd.get_dummies(bd_mca)
      ca_m = ca.fit(Z)

      tb_col = ['black','gray','brown','palegreen','teal','turquoise','yellow','purple','magenta','pink']
      df = pd.DataFrame(ca_m.row_coordinates(Z).values)
      df.columns= ['x','y','z']
      df['year'] = list(time_index.year)
      df['color'] = pd.Series(tb_col)[(df['year']-2013).values].values
      
      dc = pd.DataFrame(ca_m.column_coordinates(Z))
      dc.columns= ['x','y','z']  

      return dc,df

def MCA_plot(fig, score_type,dc,df,dim):

  lv_name = ['Bajo','Medio','Alto']
  cl_ax = ['red','orange','green']
  if score_type == 'col':
    fig.add_trace(go.Scatter(
              legendrank=2,
              legend="legend1",
              x=dc[dim[0]][:3],
              y=dc[dim[1]][:3],
              mode="text",
              name="Exp. Hurst",
              text=["P-", "EMH", "P+"],
              textposition="bottom center",
            textfont={"color": 'blue',"size": 20}
                ))

    k=0
    for ax in lv_name:
              ax_ = [True if ax in name else False for name in list(dc.index[3:])]
              fig.add_trace(go.Scatter(
                  legendrank=1,
                  legend="legend1",
                  x=dc[dim[0]][3:][ax_],
                  y=dc[dim[1]][3:][ax_],
                  mode="text",
                  name="Nivel: "+ax,
                  text=[ax_l.replace('_'+ax,'') for ax_l in list(dc.index[3:][ax_])],
                  textposition="bottom center",
                  marker_symbol = 'diamond',
                  marker_size=10,
                  marker={"color":cl_ax[k]},
                  textfont={"color":cl_ax[k],"size": 18}
              ))
              k=k+1
    fig.update_layout(legend1=dict(orientation="h",entrywidth=150,xanchor="center",x=0.5,
                                   itemsizing = 'constant',title = 'Indicadores',
                                font=dict(size=18,color="black"),))

  if score_type == 'row':
      df = df.astype({"year": str})
      for year in range(2013,2023):
          dataset_by_year = df[df['year']==str(year)]
          fig.add_trace(go.Scatter(
                legend="legend2",
                x=dataset_by_year[dim[0]],
                y=dataset_by_year[dim[1]],
                mode="markers",
                name=str(year),
                marker={"color":dataset_by_year['color'],'size':8},
                text=dataset_by_year['year'],
                textposition="bottom center",
                textfont={"color":dataset_by_year['color'],"size": 9}
            ))
      fig.update_layout(legend2=dict(orientation="v",yanchor="middle",y=0.5,bgcolor='white',
                                    itemsizing = 'constant',
                                     title ='Semanas',
                                    font=dict(size=18,color="black"),))

  fig.update_layout(showlegend=True,template='plotly_white', title="MCA: "+str(10+k_vecinos)+" semanas",
          xaxis=dict(showgrid=True,showline=False,showticklabels=True,linewidth=3,
          linecolor='rgb(204, 204, 204)',ticks='outside',
          tickfont=dict(family='Arial',size=15,color='rgb(82, 82, 82)',),
          ),
      yaxis=dict(showgrid=True,showline=False,showticklabels=True, linewidth=2,nticks=10,exponentformat = 'none',
          linecolor='rgb(204, 204, 204)',ticks='outside',
          tickfont=dict(family='Arial',size=15,color='rgb(82, 82, 82)'),
          titlefont=dict(family='Arial',size=20,color='blue'),
                ),)
  return fig

def make_matrix_dist(M_coef,col_names):
    fig = px.imshow(M_coef,x=col_names,color_continuous_scale='Purples', origin='lower',
                y=list(np.arange(10,101)))
    
    fig.update_xaxes(visible=True, fixedrange=False)
    fig.update_yaxes(visible=True, fixedrange=False)

    fig.update_layout(
          title="",
          yaxis_title='Escala TS-LHE',
          xaxis_title='Indicadores de noticias',
          legend_title="",
          title_font_color="red",
          legend_title_font_color="green",
          xaxis=dict(showgrid=False,showline=True,showticklabels=True,linewidth=3,nticks=10,
              linecolor='rgb(204, 204, 204)',ticks='outside',
              tickfont=dict(family='Arial',size=15,color='rgb(82, 82, 82)',),
              titlefont=dict(family='Arial',size=18,color='black',)
              ),
          yaxis=dict(showgrid=True,showline=False,showticklabels=True, linewidth=2,nticks=10,exponentformat = 'none',
              linecolor='rgb(204, 204, 204)',ticks='outside',
              tickfont=dict(family='Arial',size=15,color='rgb(82, 82, 82)'),
              titlefont=dict(family='Arial',size=18,color='black',)
                    ),
          width=1000,
          height=400,
          margin=dict(l=10,r=10,t=10,b=10),
          showlegend=False,
          plot_bgcolor='white')
    
    return fig

def make_mca_plot(dc,df,k_vecinos,dim):
   
    fig = go.Figure()
    fig= MCA_plot(fig,'row',dc,df,dim)
    fig = MCA_plot(fig,'col',dc,df,dim)

    fig.update_layout(template='plotly_white', title="MCA: "+str(10+k_vecinos)+" semanas")
    fig.update_layout(width=1000,height=500)
    
    return fig

#######################
# Page configuration
st.set_page_config(
    page_title="Resultados MCA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)
#######################
type_token = 'mean'
dim = ['x','y']

ca = prince.CA(
    n_components=3,
    n_iter=3,
    copy=True,
    check_input=True,
    engine='sklearn',
    random_state=20
)

#####################
# Sidebar
with st.sidebar:
    st.title('Eficiencia de la BMV con base en indicadores de noticias')   
    
    selected_criterio = 'DTW'
    selected_criterio = st.selectbox('Selecciona un criterio', ['DTW','CG','CSE']) 
    
    selected_norm = 'Núm. noticias'
    selected_norm = st.selectbox('Selecciona la estandarización', ['Núm. noticias','MAX-MIN']) 
         
    st.write('''- :orange[**Información**]:  A continuación se muestran los resultados de la búsqueda de relaciones entre la eficiencia del mercado y los indicadores de noticias. Para examinar la relación entre los indicadores de noticias y los exponentes de Hurst se calcularon medidas para buscar similitudes o sincronía entre las series y, además, se usó MCA para obtener una representación de las categorías de los indicadores.''') 
    
    if selected_norm == 'Núm. noticias':
        type_size = 'ts'
        indices = pd.read_csv('./Indicadores_num_not.csv',index_col = 'Semana',parse_dates=True)
        if selected_criterio == 'DTW':
            M_coef = pd.read_csv('./Matriz_DTW.csv',index_col = 'Escala')
        elif selected_criterio == 'CG':
            M_coef = pd.read_csv('./Matriz_CG.csv',index_col = 'Escala')
        else:
            M_coef = pd.read_csv('./Matriz_CSE.csv',index_col = 'Escala')

    else:
        type_size = 'mx'
        indices = pd.read_csv('./Indicadores_max_min.csv',index_col = 'Semana',parse_dates=True)
        if selected_criterio == 'DTW':
            M_coef = pd.read_csv('./Matriz_DTW_MaxMin.csv',index_col = 'Escala')
        elif selected_criterio == 'CG':
            M_coef = pd.read_csv('./Matriz_CG_MaxMin.csv',index_col = 'Escala')
        else:
            M_coef = pd.read_csv('./Matriz_CSE_MaxMin.csv',index_col = 'Escala')

#######################
#Data import

#Exportación de datos
ipc_return = pd.read_csv('./Retornos_IPC.csv',index_col = 'Semana',parse_dates = True)
H_matrix = pd.read_csv('./Exponentes_Hurst.csv',index_col = 'Escala')
if selected_criterio == 'CSE':
    k_vecinos = np.argmin(np.mean(M_coef,axis=1))
else:
    k_vecinos = np.argmax(np.mean(M_coef,axis=1))
dc, df = MCA_score(indices,k_vecinos,ipc_return.index)

##################################################3
# Dashboard Main Panel
col = st.columns((0.3, 0.7), gap='small')
  
with col[0]:
    st.write('''- :orange[**Metodología**]:''') 
    st.write(''' La estrategia de trabajo comenzó con la adquisición de los valores del IPC para analizar la hipótesis de la eficiencia del mercado usando el algoritmo TS-LHE a la serie de retornos del IPC. Después se aplicó la modelación de tópicos y el análisis de sentimiento para generar indicadores de noticias. En la siguiente tabla se proporciona una recopilación de la información de los indicadores de texto y la simbología del indicador en las gráficas posteriores.''')
    st.image('./tabla.png')
    
    with st.expander('', expanded=True):
            st.write('''- :orange[**1. Estandarizar los indicadores de noticias.**]: En la primera metodología, se dividió el valor del indicador entre el número de noticias recopiladas en la semana de la sección correspondiente. En el segundo enfoque, se aplicó la siguiente fórmula a cada indicador $\{w_{it}\}$: 
    $\dfrac{w_{it}-max(w_i)}{max(w_i)-min(w_i)}$ ''') 
    with st.expander('', expanded=False):
            st.write('''- :orange[**2. Encontrar escala óptima entre los indicadores de noticias y la serie de exponentes de Hurst.**]: Se utilizaron tres métodos: la distancia de deformación dinámica del tiempo (DTW), la causalidad de Granger (GC) y la entropía entre muestras (CSE). Para cada método, se creó una matriz de resultados con los indicadores en las columnas y se calculó el promedio de las mediciones por fila, seleccionando la fila con el valor promedio más alto (o el más bajo en el caso de la entropía) como la escala óptima.''') 
    with st.expander('', expanded=False):
            st.write('''- :orange[**3. Categorizar los exponentes de Hurst en persistencia P+, eficiencia EMH y anti persistencia P-.**]: Los exponentes de Hurst se categorizaron en tres grupos: 
    - Persistente (P+) si el exponente de la semana es mayor a $0.5+\hat{\sigma}(H)$
    - Eficiente (EMH) si encuentra entre $[0.5-\hat{\sigma}(H), 0.5+\hat{\sigma}(H)]$ 
    - Anti-persistente (P-) si es menor a $0.5-\hat{\sigma}(H)$ 
    
    donde $ \hat{\sigma}(H)$ es aproximadamente $\dfrac{1}{e\sqrt[3]{N}}$ ''') 
    with st.expander('', expanded=False):
            st.write('''- :orange[**4. Categorizar los indicadores de noticias en alto, intermedio y bajo.**]:  Los indicadores de noticias se categorizaron en tres niveles: bajo si el valor es menor a 0.25, medio si es entre 0.25 y 0.5 y alto si es mayor a 0.5. ''') 
    with st.expander('', expanded=False):
            st.write('''- :orange[**5. Crear la matriz indicadora de los datos**]: Se creó una tabla con las 521 semanas de estudio en las filas y los indicadores de noticias junto con el exponente de Hurst categorizado se colocaron en las columnas. Luego, esta tabla se transformó en una matriz indicadora. ''') 
    with st.expander('', expanded=False):
            st.write('''- :orange[**6. Aplicar el análisis de correspondencia múltiple en la matriz indicadora.**]: Se aplicó el análisis de correspondencia múltiple (MCA) sobre la matriz indicadora para generar una representación gráfica de las filas y columnas.''')                
    
with col[1]:
    
    st.markdown('#### Relación entre exponentes e indicadores')
    treemap = make_matrix_dist(M_coef,indices.columns)
    st.plotly_chart(treemap, use_container_width=True)  

    st.markdown('#### Representación con MCA')
    pie = make_mca_plot(dc,df,k_vecinos,dim)
    st.plotly_chart(pie, use_container_width=True)