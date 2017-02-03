import os
import errno

def select_gpu():
    print 'Automatic gpu selection.'
    try:
        import pynvml
        pynvml.nvmlInit()
        numOfGPUs = int(pynvml.nvmlDeviceGetCount())
        res = []
        for i in range(0, numOfGPUs):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            try:
                memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = memInfo.used / 2 ** 20
                mem_free = memInfo.free / 2 ** 20
            except pynvml.NVMLError as err:
                error = pynvml.handleError(err)
                mem_total = error
                mem_used = error
                mem_free = error
            print i, mem_free
            res.append((i, mem_free))

            # NOTE: You can also iterate processes running on this gpu.

            # procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            #
            # for p in procs:
            #     try:
            #         name = str(pynvml.nvmlSystemGetProcessName(p.pid))
            #         print name, p.usedGpuMemory / 2 ** 20
            #     except pynvml.NVMLError as err:
            #         if (err.value == pynvml.NVML_ERROR_NOT_FOUND):
            #             # probably went away
            #             continue
            #         else:
            #             name = handleError(err)
        gpu_id, free = sorted(res, key=lambda a: a[1])[-1]
        MINIMAL_FREE_MEMORY_MB = 1000

        if free < MINIMAL_FREE_MEMORY_MB:
            raise RuntimeError('No free gpu!')

        print 'Selecting device={id}, with {free} mbs of free memory'.format(id=gpu_id, free=free)
        return gpu_id

    except pynvml.NVMLError, err:
        print "Failed to initialize NVML: ", err
        print "Exiting..."
        os._exit(1)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
