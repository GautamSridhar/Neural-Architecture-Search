from collections import namedtuple

Genotype = namedtuple('Genotype', 'gene')

PRIMITIVES = [
    'Identity',
    'LinearReLU_2',
    'LinearReLU_4',
    'LinearReLU_8',
    'LinearReLU_16',
    'LinearTanh_2',
    'LinearTanh_4',
    'LinearTanh_8',
    'LinearTanh_16',
]