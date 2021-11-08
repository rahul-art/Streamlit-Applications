import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("WebRTC demo")
    delayed_echo_page = "Delayed echo (sendrecv)"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            delayed_echo_page
        ],
    )
    st.subheader(app_mode)
    if app_mode == delayed_echo_page:
        app_delayed_echo()

    st.sidebar.markdown(
        """
---
<a href="https://www.buymeacoffee.com/whitphx" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" height="50" ></a>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")

def app_delayed_echo():
    #DEFAULT_DELAY = 1.0

    class VideoProcessor(VideoProcessorBase):
        #delay = DEFAULT_DELAY

        async def recv_queued(self, frames: List[av.VideoFrame]) -> List[av.VideoFrame]:
            #logger.debug("Delay:", self.delay)
            #await asyncio.sleep(self.delay)
            return frames

    class AudioProcessor(AudioProcessorBase):
        #delay = DEFAULT_DELAY

        async def recv_queued(self, frames: List[av.AudioFrame]) -> List[av.AudioFrame]:
            #await asyncio.sleep(self.delay)
            return frames

    webrtc_ctx = webrtc_streamer(
        key="delay",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()

