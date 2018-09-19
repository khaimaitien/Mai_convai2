#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for f1 metric
Key-Value Memory Net model trained on convai2:self
"""

from parlai.core.build_data import download_models
from projects.convai2.eval_f1 import setup_args, eval_f1

if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        model='parlai.agents.k_model.k_agent:KMethod',
        numthreads=80,
    )
    opt = parser.parse_args(print_args=False)
    eval_f1(parser, print_parser=parser)
