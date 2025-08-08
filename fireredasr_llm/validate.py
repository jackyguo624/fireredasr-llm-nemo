# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For kaldi wav.ark io
# ```bash
# git clone https://github.com/jackyguo624/Sequential.git
# cd Sequential && pip install .
# ```
#
# from sequential.data.lhotse.audio_backend import KaldiioBackend
# from lhotse.audio.backend import set_current_audio_backend
# 
# set_current_audio_backend(KaldiioBackend())


from fireredasr_llm.collections.speechllm.recipes import speech_to_text_llm_validate
from nemo.core.config import hydra_runner


@hydra_runner(config_path="./conf", config_name="salm-qwen2-7b_fc_fc_valid")
def main(cfg):
    """main function for running validation."""
    return speech_to_text_llm_validate(cfg)


if __name__ == "__main__":
    main()
