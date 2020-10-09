# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, too-many-arguments, too-many-nested-blocks
"""Interpolate operator"""

from tvm.te import hybrid

"""
@hybrid.script
def _interpolate_1d(x, xp, fp):

    lenx = x.shape[0]
    lenxp = xp.shape[0]
    out = output_tensor(x.shape, x.dtype)
    minxp = xp[0]
    maxxp = xp[lenxp-1]

    #Assume x, xp, fp are sorted in increasing order

    for i in range(lenx-1):
        if x[i] <= minxp:
            out[i] = fp[0]
        elif x[i] >= maxxp:
            out[i] = fp[lenxp-1]
        else:
            #Is i-1 correct here?
            #Not just based on i, need left and right...
            out[i] = fp[i-1] + (i - x[i-1])*((fp[i]-fp[i-1])/(x[i]-x[i-1]))

    return out

    
    #Test basic
    #out = output_tensor(x.shape, x.dtype)
    #out[0]  = 1.0

    #return out
"""

@hybrid.script
def _interpolate_1d(x, xp, fp):

    lenx = x.shape[0]
    lenxp = xp.shape[0]
    minxp = xp[0]
    maxxp = xp[lenxp-1]

    out = output_tensor(x.shape, x.dtype)

    for i in range(lenx):
        for j in range(lenxp):
            if x[i] <= minxp:
                out[i] = fp[0]
            elif x[i] >= maxxp:
                out[i] = fp[lenxp-1]
            elif j>0 and x[i] >= xp[j-1] and x[i] < xp[j]:
                out[i] = fp[j-1] + (x[i] - xp[j-1])*((fp[j]-fp[j-1])/(xp[j]-xp[j-1]))

    #print('in interp out')
    #print(out)
    return out

def interpolate(x, xp, fp):

    if len(x.shape) > 1:
        raise ValueError("Interpolate only supports 1d")
    else:
        return _interpolate_1d(x, xp, fp)