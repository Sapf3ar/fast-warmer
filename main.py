import sys
from datetime import datetime
from random import sample
from efffects import get_lapunov,get_random_walk 
from relevant import CandidatesSelector
import pandas as pd
import streamlit as st
from loguru import logger
from pydantic import BaseModel
import random
import cv2


@st.cache_data
def get_drawing(**kwargs):
    if random.random() >0.5:
        return get_random_walk(**kwargs)
    else:
        return get_lapunov(**kwargs)
    

st.set_page_config(layout="wide",
                   page_title="Fast-Warmer",
                   page_icon=":ocean:",
    menu_items={"Report a Bug":None,
                "About":None,
                "Get help":None})
logger.add(sys.stdout, level="INFO")
if "total_likes" not in st.session_state:
    st.session_state['total_likes'] = 0
    for i in range(0, 5):
        col_pos = "l"
        st.session_state[f"like_{col_pos}_{i}"] = None
    for i in range(5, 10):
        col_pos = "r"
        st.session_state[f"like_{col_pos}_{i}"] = None

class Item(BaseModel):
    name: str
    date: datetime
    description: str | None
    category:str
    video_id: str
    interaction: int = 0


def get_recs():
    
    if st.session_state.total_likes ==0:

        video_items:list[Item] = CandidatesSelector().get_n_candidates(st.session_state['data'], n=10)
        st.session_state["items"] = video_items
        for key in st.session_state.keys():
            if str(key).startswith("like"):
                st.session_state[key] = None
    else:
        ids = list(range(len(st.session_state["data"])))
        rand_ids = sample(ids, k=10)
        samples = st.session_state["data"].loc[
            st.session_state["data"].index[rand_ids]
        ]
        video_items = []
        for id_, row in samples.iterrows():
            video_items.append(
                Item(
                    name=row.title,
                    description=row.description,
                    video_id=row.video_id,
                    date=row.v_pub_datetime,
                )
            )

        st.session_state["items"] = video_items
        for key in st.session_state.keys():
            if str(key).startswith("like"):
                st.session_state[key] = None


col1_t, _, _, _, col2_b = st.columns([0.3, 0.2, 0.2, 0.15, 0.15])

col1_t.markdown(f"**You have made :rainbow[{st.session_state['total_likes']} reactions!]**")
col2_b.button(type='primary',
              label="Обновить рекомендации",
              on_click=get_recs,
              key='rec_button')

@st.cache_data(persist="disk", show_spinner=True)
def load_data(path:str ='top_200_per_category.csv') -> pd.DataFrame:

    df = pd.read_csv(path, parse_dates=['v_pub_datetime'])
    return df




def _init_state():
    st.session_state["data"] = load_data()
    # load_data()


_init_state()


def update_feedback(*args, **kwargs):
    if not kwargs:
        return
    pos = kwargs['pos']
    pos_id = int(pos.split("_")[-1])
    like_value = 1. if st.session_state[pos] else -1.
    st.session_state['items'][pos_id].interation = like_value
    st.session_state.total_likes += 1


def load_startup_recs():
    if "started" not in st.session_state:
        st.session_state.started = True
        get_recs()


load_startup_recs()


def get_video(pos: int) -> Item:
    if st.session_state['items'] is None:
        return Item(name="placeholder" + str(pos),
                    date=datetime.now(),
                    video_id="0",
                    description=" ")
    else:
        return st.session_state['items'][pos]


_, col1, _, col2, _ = st.columns(gap="medium", spec=[0.15, 0.3, 0.1, 0.3, 0.15])
like_options = (":thumbsup:",":thumbsdown:")
for i in range(0, 5):
    col_pos = "l"
    video_item:Item = get_video(i)
    hash_input = "".join(list(set(video_item.name)))
    col1.image(get_drawing(in_str=hash_input),use_column_width="auto")
    exp = col1.expander(label="**"+ video_item.name + "**", icon=':material/sort:')
    video_name = "**"+ video_item.name + "**"

    exp = col1.expander(label=video_name, icon=':material/sort:')
    exp.markdown(body=video_item.description, unsafe_allow_html=True) 

    pos =f"like_{col_pos}_{i}"
    col1.radio(
        label=".",  
        options=[0, 1] ,
        format_func=lambda x: like_options[x],
        index=st.session_state[pos],
        key=pos,
        on_change=update_feedback,
        label_visibility='hidden',
        horizontal=True,
        kwargs={"pos":pos},
    )


for i in range(5, 10):
    col_pos = "r"
    video_item:Item = get_video(i)
    hash_input = "".join(list(set(video_item.name)))
    col2.image(get_drawing(in_str=hash_input),use_column_width="auto")
    video_name = "**"+ video_item.name + "**"
    exp = col2.expander(label=video_name, icon=':material/sort:')
    exp.markdown(body=video_item.description,unsafe_allow_html=True) 
    pos =f"like_{col_pos}_{i}"
    col2.radio(
        label=".",  
        options=[0, 1] ,
        format_func=lambda x: like_options[x],
        index=st.session_state[pos],
        key=pos,
        on_change=update_feedback,
        label_visibility='hidden',
        horizontal=True,
        kwargs={"pos":pos},
    )
