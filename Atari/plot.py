import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def smoothTriangle(data, degree, dropVals=False):
    triangle=np.array(list(range(degree)) + [degree] + list(range(degree)[::-1])) + 1
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals:
        return smoothed
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

    
def extract_file(filename):

    f = open(filename, "r")
    values = []
    count=0
    for line in f:
        count+=1
        values.append((count,float(line[68:].replace(":","").strip())))
        if count ==505:
            break
    return values


if __name__ =="__main__":

    PLOT_TITLE = "DQN"
    

    DQN = pd.DataFrame(extract_file("DQNLOG.txt"),columns=['Episode', 'Reward'])
    
    DDQN = pd.DataFrame(extract_file("DDQNLOG.txt"),columns=['Episode', 'Reward'])

    DuelingDQN = pd.DataFrame(extract_file("DuelingDQNLOG.txt"),columns=['Episode', 'Reward'])

    DuelingDDQN = pd.DataFrame(extract_file("DuelingDDQNLOG.txt"),columns=['Episode', 'Reward'])

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DQN["Episode"], y=smoothTriangle(DQN["Reward"],7),
                        mode='lines',
                        name='DQN'))
    
    # fig.add_trace(go.Scatter(x=DDQN["Episode"], y=smoothTriangle(DDQN["Reward"],7),
    #                     mode='lines',
    #                     name='DDQN',
    #                     line=dict(color="#FF0000")))

    # fig.add_trace(go.Scatter(x=DuelingDQN["Episode"], y=smoothTriangle(DuelingDQN["Reward"],7),
    #                     mode='lines',
    #                     name='Dueling DQN',
    #                     line=dict(color="#64e747")))

    # fig.add_trace(go.Scatter(x=DuelingDDQN["Episode"], y=smoothTriangle(DuelingDDQN["Reward"],7),
    #                     mode='lines',
    #                     name='Dueling DDQN',
    #                     line=dict(color="#9041da")))

    
    fig.update_layout(
    title=PLOT_TITLE,
    xaxis_title="Episodes",
    yaxis_title="Average Reward",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    ),
    showlegend=True
)
    fig.show()
