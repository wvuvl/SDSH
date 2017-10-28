# Copyright 2017 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple logging utils"""

import sys


class Log:
    """Save all prints to file, while keeping them in the console too"""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        class __Tee:
            def write(self, *args, **kwargs):
                self.out1.write(*args, **kwargs)
                self.out2.write(*args, **kwargs)
                self.out1.flush()
                self.out2.flush()

            def __init__(self, out1, out2):
                self.out1 = out1
                self.out2 = out2

        self.log_file = open(self.path, "a")
        self.stdout = sys.stdout
        sys.stdout = __Tee(self.log_file, sys.stdout)

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        self.log_file.close()
        return False


def __main():
    with Log("aaa.txt"):
        print("AAAAAA")
        print("AAAAAA {0}".format(1))

if __name__ == '__main__':
    __main()
    print("BBBBBBB")

