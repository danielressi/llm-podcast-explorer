from plotly import graph_objects as go
import numpy as np

def play_button(frame_duration, transition_duration):
    return dict(label="Play",                                                
                method="animate",
                args=[
                        None,
                        {
                            "frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, 
                            "mode": "immediate",
                            "transition": {"duration": transition_duration, "easing": "quad-out"},
                        }
                    ]
                )


def stop_button():
    return dict(
        label="Stop",
        method="animate",

        args=[
            [], 
            dict(
                frame=dict(duration=0, redraw=True),
                mode="immediate"
            )
        ]
    )

def zoom_out_animation(fig, selection_data, delta_init=0.1, frame_duration=40000, transition_duration=40000):
    
    fig.update_layout(xaxis_range=[selection_data["x"] - delta_init, selection_data["x"] + delta_init], 
                        yaxis_range=[selection_data["y"]  - delta_init, selection_data["y"]  + delta_init])
    
    # Define zoom-out animation frames
    zoom_frames = [
        go.Frame(
            layout=dict(
                xaxis=dict(range=[selection_data["x"] - delta, selection_data["x"] + delta]),
                yaxis=dict(range=[selection_data["y"]  - delta, selection_data["y"]  + delta]),
            )
        )
        for delta in [5, delta_init]
    ]

    fig.frames = zoom_frames
    fig.update_layout(updatemenus=[dict(type="buttons", 
                                        showactive=False,
                                        x=1.1,
                                        xanchor="left",  
                                        #yanchor="bottom",
                                        buttons=[
                                            play_button(frame_duration, transition_duration),
                                            #stop_button(),
                                            ]
                                            )])