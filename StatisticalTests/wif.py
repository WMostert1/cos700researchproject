from Queue import Queue
import time
from threading import Thread, Lock
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import random


class AtomicInteger:
    def __init__(self, value=0):
        self._value = value
        self._lock = Lock()

    def inc(self):
        with self._lock:
            self._value += 1
            return self._value

    def dec(self):
        with self._lock:
            self._value -= 1
            return self._value

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = v
            return self._value


def run(number_of_customers, num_worker_threads, worker_low_ms, worker_high_ms, filename, new_heuristic=False):
    wif = [AtomicInteger() for x in range(num_worker_threads)]

    q = Queue(32)

    wif_rejections = AtomicInteger()

    q_sizes = []

    def worker():
        while True:
            customer = q.get()

            customer_wif = wif[customer]
            customer_wif.inc()

            current_queue_size = q.qsize()
            q_sizes.append(current_queue_size)

            if new_heuristic:
                if customer_wif.value > 5 and current_queue_size > 25:
                    wif_rejections.inc()
                    q.task_done()
                    continue
            elif customer_wif.value > 5:
                wif_rejections.inc()
                q.task_done()
                continue

            time.sleep(random.uniform(worker_low_ms, worker_high_ms))

            customer_wif.dec()
            q.task_done()

    def producer(number_of_requests, batch, customer, wait):
        num_reqs = 0
        while num_reqs < number_of_requests:
            for a in range(batch):
                q.put(customer)
                num_reqs += 1
            time.sleep(wait)

    for i in range(num_worker_threads):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    producer_threads = [None for c in range(number_of_customers)]
    for c in range(number_of_customers):
        t = Thread(target=producer(1000, 8, c, 0.05))
        producer_threads[c] = t
        t.daemon = True
        t.start()

    for pt in producer_threads:
        pt.join()

    q.join()  # block until all tasks are done

    print(wif_rejections.value)

    title = "Worker min max latency (" + str(worker_low_ms) + ", " + str(worker_high_ms) + "), WIF Rejections = " + str(
        wif_rejections.value)

    density = gaussian_kde(q_sizes)
    xs = np.linspace(0, 32, 5000)
    density.covariance_factor = lambda: .5
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.xlabel("Queue Length")
    plt.ylabel("Density")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


run(5, 8, 0.5, 0.05, "out/WIF-LowQueue.png")
run(5, 8, 0.05, 0.09, "out/WIF-HighQueue.png")

run(5, 8, 0.005, 0.05, "out/WIF-QueueLengthHeuristicLowQueue.png", True)
run(5, 8, 0.05, 0.09, "out/WIF-QueueLengthHeuristicHighQueue.png", True)
