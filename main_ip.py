from pycgp import CGP, CGPJunkyDNA, CGPES, Evaluator, MaskEvaluator
from pycgp.ipfunctions import *
import pandas as pd
import sys
import time





def evolveMask(col=30, row=1, nb_ind=5, mutation_rate_nodes=0.15, mutation_rate_outputs=0.3,
              n_cpus=1, n_it=2000 , genome=None):
    library = build_funcLib()

    dirname = '/Users/cussat/Recherche/Projects/CGP/Datasets/odonto/Dataset_2/'
    tag = 'dent'
    dataset_name = 'dataset_'+tag+'.csv'

#    dirname = '/Users/cussat/Recherche/Projects/CGP/Datasets/melanoma/'
#    dataset_name = 'dataset.csv'
#    tag = 'featfoncs'

#    dirname = '/Users/cussat/Recherche/Projects/CGP/Datasets/Histo/'
#    dataset_name = 'dataset_context.csv'
#    tag = 'context'


    e = MaskEvaluator(dirname = dirname,
					  dataset_name = dataset_name,
                      display_dataset=False,
                      resize = 0.25,
                      include_hsv = True,
                      include_hed = False,
                      number_of_evaluated_images=-1)
    
    if genome is None:
#        cgpFather = CGP.random(1, 1, col, row, library, 1.0, False, 256, 0, 255, in_image[0].shape, 'uint8')
        cgpFather = CGP.random(num_inputs=e.n_inputs, num_outputs=e.n_outputs, num_cols=col, num_rows=row, library=library, recurrency_distance=1.0, recursive=False, const_min=0, const_max=255, input_shape=e.input_channels[0][0].shape, dtype='uint8')
    else:
        cgpFather = CGP.load_from_file(genome, library)
    es = CGPES(nb_ind, mutation_rate_nodes, mutation_rate_outputs, cgpFather, e, dirname+'evos/run_'+str(round(time.time() * 1000000))+'_'+tag, n_cpus)
    es.run(n_it)

    es.father.to_function_string(['ch_'+str(i) for i in range(e.n_inputs)], ['mask_'+str(i) for i in range(e.n_outputs)])
#    es.father.to_dot(folder_name+'/best.dot', ['x'], ['y'])
#    os.system('dot -Tpdf ' + folder_name+'/best.dot' + ' -o ' + folder_name+'/best.pdf')

    #es.father.graph_created = False
    e.evaluate(es.father, 0, True)


def load(file_name):
    print('loading ' + file_name)
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    e = MaskEvaluator()
    print(e.evaluate(c, 0, displayTrace=True))
   
def toDot(file_name, out_name):
    print('Exporting ' + file_name + ' in dot ' + out_name + '.dot')
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_dot(out_name+'.dot', ['x'], ['y'])
    print('Converting dot file into pdf in ' + out_name + '.pdf')
    os.system('dot -Tpdf ' + out_name + '.dot' + ' -o ' + out_name + '.pdf')

def displayFunctions(file_name):
    library = build_funcLib()
    c = CGP.load_from_file(file_name, library)
    c.to_function_string(['x'], ['y'])


if __name__ == '__main__':

    print (sys.argv)
    if len(sys.argv) == 2:
        evolveMask()
    elif len(sys.argv)==1:
        #print('Evaluating '+sys.argv[1])

        library = build_funcLib()
    
        dirname = '/Users/cussat/Recherche/Projects/CGP/Datasets/odonto/Dataset_2/'
        dataset_name = 'dataset_dent.csv'
        subset = 'training'
    
        e = MaskEvaluator(dirname = dirname,
						  dataset_name = dataset_name,
                          display_dataset=False,
                          resize = 0.25,
                          include_hsv = True,
                          subset = subset)

        cgp = CGP.load_from_file(dirname+'evos/run_1690404750231671_dent/cgp_genome_1974_0.7828176903220382.txt', library, input_shape=e.input_channels[0][0].shape, dtype='uint8')
#        cgp = CGP.random(num_inputs=e.n_inputs, num_outputs=e.n_outputs, num_cols=30, num_rows=1,
#                               library=library,
#                               recurrency_distance=1.0, recursive=False, const_min=0, const_max=255,
#                               input_shape=e.input_channels[0][0].shape, dtype='uint8')

        e.evaluate(cgp, 0, displayTrace=True)
        exit(0)


        cgp.create_graph()


        fits = []
        for i in range(cgp.num_inputs, cgp.num_inputs+cgp.num_cols):
            cgp_mutated = cgp.clone()
            if i != cgp.genome[-1] and i not in cgp.nodes_used:
                cgp_mutated.genome[-1] = i
                cgp_mutated.graph_created = False
                fits.append(e.evaluate(cgp_mutated, 0, displayTrace=True))
        fits = np.array(fits)
        print(str(fits.mean())+'\t'+
              str(fits.std())+'\t'+
              str(fits.max()) + '\t' +
              str(fits.min()))


# TODO List
# revoir la lecture des fichiers ROI: 1 seul mask sort de la liste (voir background odonto)
#    - voir: https://github.com/jayunruh/napari_jroitools
# lecture du dataset en particulier en test
# regarder fourrier