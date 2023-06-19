from neuron import Neuron
from som import SOM, data_supplier_uniform_box, data_supplier_non_uniform_boxes, data_supplier_non_uniform_scatter, \
    data_supplier_uniform_donut, data_supplier_non_uniform_donut
from multiprocessing import Process

def worker_box(radius_factor, iterations, neurons):
    s = SOM(initial_radius_factor=radius_factor, max_iterations=iterations, neurons_count=neurons)
    s.init_line_of_neurons(0.0, 0.0, 1.0, 1.0)
    s.train(data_supplier_uniform_box)
    s.create_gif()

def worker_cicrcle(radius_factor, iterations, neurons):
    s = SOM(initial_radius_factor=radius_factor, max_iterations=iterations, neurons_count=neurons)
    s.init_circle_of_neurons(16.0, 16.0)
    s.train(data_supplier_non_uniform_donut)
    s.create_gif()

def main():
        workers = []
        for radius_factor in [2, 4, 6, 10]:
            for neurons in [300]:
                p = Process(target=worker_cicrcle, args=(radius_factor, 500, neurons))
                p.daemon = True
                p.start()
                workers.append(p)
        [x.join() for x in workers]


if __name__ == "__main__":
    main()