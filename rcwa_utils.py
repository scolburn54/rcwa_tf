import tensorflow as tf
import numpy as np

def fftshift_tf(matrix):
    '''
    Performs a fftshift operation on a given matrix with odd dimensions.
    Args:
        matrix: A `tf.Tensor` of dtype `float` and shape `(batchSize, pixelsX, pixelsY, layers, P * Q, P * Q)` of
                Fourier components with the DC component at position [0, 0].
    Returns:
        A `tf.Tensor` of dtype `float` and shape `(batchSize, pixelsX, pixelsY, layers, P * Q, P * Q)`
        of Fourier components with the DC component centered.
    '''
    
    _, _, _, _, Nx, _ = matrix.shape
    shiftAmt = int(np.floor(Nx / 2.0))
    shifted_once = tf.roll(matrix, shift=shiftAmt, axis = 4) 
    return tf.roll(shifted_once, shift=shiftAmt, axis = 5)
    
def convmat(A, P, Q):
    '''
    This function computes a convolution matrix for a real space matrix `A` that represents
    either a relative permittivity or permeability distribution for a set of pixels, layers, and batch.
    Args:
        A: A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX, pixelsY, Nlayers, Nx, Ny)` 
           specifying real space values on a Cartesian grid.
        P: A positive and odd `int` specifying the number of spatial harmonics along `T1`.
        Q: A positive and odd `int` specifying the number of spatial harmonics along `T2`.
    Returns:
        A `tf.Tensor` of dtype `complex` and shape `(batchSize, pixelsX, pixelsY, Nlayers, P * Q, P * Q)`
        representing a stack of convolution matrices based on `A`.   
    '''
    
    # Determine the shape of A
    batchSize, pixelsX, pixelsY, Nlayers, Nx, Ny = A.shape

    # Compute indices of spatial harmonics
    NH = P * Q # total number
    p_max = np.floor(P / 2.0)
    q_max = np.floor(P / 2.0)
    p = np.linspace(-p_max, p_max, P) # indices along T1
    q = np.linspace(-q_max, q_max, Q) # indices along T2
    
    # Compute array indices of center harmonic
    p0 = int(np.floor(Nx / 2))
    q0 = int(np.floor(Ny / 2))

    # Fourier transform the real space distributions
    A = fftshift_tf(tf.signal.fft2d(A)) / (Nx * Ny)

    # Build the matrix
    firstCoeff = True
    for qrow in range(Q):
        for prow in range(P):
            for qcol in range(Q):
                for pcol in range(P):
                    pfft = int(p[prow] - p[pcol])
                    qfft = int(q[qrow] - q[qcol])

                    # Sequentially concatenate Fourier coefficients
                    value = A[:, :, :, :, p0 + pfft, q0 + qfft]
                    value = value[:, :, :, :, tf.newaxis, tf.newaxis]
                    if firstCoeff:
                        firstCoeff = False
                        C = value
                    else:
                        C = tf.concat([C, value], axis = 5)
                        
    # Reshape the coefficients tensor into a stack of convolution matrices
    convMatrixShape = (batchSize, pixelsX, pixelsY, Nlayers, P * Q, P * Q)      
    matrixStack = tf.reshape(C, shape = convMatrixShape)

    return matrixStack

def redheffer_star_product(SA, SB):
    '''
    This function computes the redheffer star product of two block matrices, which is the result
    of combining the S-parameter of two systems.
    Args:
        SA: A `dict` of `tf.Tensor` values specifying the block matrix corresponding to
           the S-parameters of a system. `SA` needs to have the keys ('S11', 'S12', 'S21', 'S22'),
           where each key maps to a `tf.Tensor` of shape `(2*NH, 2*NH)`, where NH is the total number
           of spatial harmonics.
        SA: A `dict` of `tf.Tensor` values specifying the block matrix corresponding to
           the S-parameters of a second system. `SB` needs to have the keys ('S11', 'S12', 'S21', 'S22'),
           where each key maps to a `tf.Tensor` of shape `(2*NH, 2*NH)`, where NH is the total number
           of spatial harmonics.
    Returns:
           A `dict` of `tf.Tensor` values specifying the block matrix corresponding to
           the S-parameters of the combined system. `SA` needs to have the keys ('S11', 'S12', 'S21', 'S22'),
           where each key maps to a `tf.Tensor` of shape `(2*NH, 2*NH), where NH is the total number
           of spatial harmonics.   
    '''
    # Define the identity matrix
    dim, _ = SA['S11'].shape
    I = tf.constant(np.eye(dim), dtype = tf.complex64)
    
    # Calculate S11
    S11 = tf.linalg.inv(I - tf.matmul(SB['S11'], SA['S22']))
    S11 = tf.matmul(S11, SB['S11'])
    S11 = tf.matmul(SA['S12'], S11)
    S11 = SA['S11'] + tf.matmul(S11, SA['S21'])
    
    # Calculate S12
    S12 = tf.linalg.inv(I - tf.matmul(SB['S11'], SA['S22']))
    S12 = tf.matmul(S12, SB['S12'])
    S12 = tf.matmul(SA['S12'], S12)
    
    # Calculate S21
    S21 = tf.linalg.inv(I - tf.matmul(SA['S22'], SB['S11']))
    S21 = tf.matmul(S21, SA['S21'])
    S21 = tf.matmul(SB['S21'], S21)
    
    # Calculate S22
    S22 = tf.linalg.inv(I - tf.matmul(SA['S22'], SB['S11']))
    S22 = tf.matmul(S22, SA['S22'])
    S22 = tf.matmul(SB['S21'], S22)
    S22 = SB['S22'] + tf.matmul(S22, SB['S12'])
    
    # Store S parameters in an output dictionary
    S = dict({})
    S['S11'] = S11
    S['S12'] = S12
    S['S21'] = S21
    S['S22'] = S22
    
    return S
