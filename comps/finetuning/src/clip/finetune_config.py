# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 The LLM-on-Ray Authors.

from typing import List, Optional, Union

from pydantic import BaseModel, Field, validator

from comps.cores.proto.api_protocol import FineTuningJobsRequest

class GeneralConfig(BaseModel):
    tool: str = None
    trainer: str = None
    model: str = None
    config_file: str = None
    dataset: str = None
    device: str = None

    # @validator("report_to")
    # def check_report_to(cls, v: str):
    #     assert v in ["none", "tensorboard"]
    #     return v

    @validator("tool")
    def check_task(cls, v: str):
        assert v in ["clip", "adaclip"]
        return v

class FinetuneConfig(BaseModel):
    General: GeneralConfig = GeneralConfig()


class FineTuningParams(FineTuningJobsRequest):
    # priority use FineTuningJobsRequest params
    General: GeneralConfig = GeneralConfig()

