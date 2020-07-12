import preprocess_cython
import time

if __name__ == '__main__':
    s = time.time()
    preprocess_cython.func()
    runtime = time.time() - s
    print(runtime)