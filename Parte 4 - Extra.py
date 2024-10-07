from mpi4py import MPI
import numpy as np

# Caminhos para as matrizes de entrada
matrix_paths = r"Matrizes/"
matrix_a = "Matriz5.txt"
matrix_b = "Matriz6.txt"

# Inicialização do ambiente MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Definir o tamanho do bloco
block_size = 100

# Função para ler um bloco específico de uma matriz
def read_matrix_block(file_path, row_start=None, row_end=None, col_start=None, col_end=None):
    data = np.loadtxt(file_path, skiprows=row_start, max_rows=(row_end - row_start) if row_end is not None else None)
    if col_start is not None and col_end is not None:
        return data[:, col_start:col_end]
    return data

def main():
    if rank == 0:
        A_full = np.loadtxt(matrix_paths + matrix_a)
        B_full = np.loadtxt(matrix_paths + matrix_b)
        A_shape = A_full.shape
        B_shape = B_full.shape
    else:
        A_shape = None
        B_shape = None

    # Broadcast das dimensões para todos os processos
    A_shape = comm.bcast(A_shape, root=0)
    B_shape = comm.bcast(B_shape, root=0)

    # Cálculo do número de blocos
    num_blocks_A = (A_shape[0] + block_size - 1) // block_size
    num_blocks_B = (B_shape[1] + block_size - 1) // block_size

    if rank != 0:
        requests = []
        
        # Processos workers processam blocos de linhas
        for block_row_A in range(rank - 1, num_blocks_A, size - 1):
            row_start_A = block_row_A * block_size
            row_end_A = min(row_start_A + block_size, A_shape[0])
            
            # Buffer para receber o bloco de A
            A_block = np.empty((row_end_A - row_start_A, A_shape[1]), dtype=np.float64)
            
            # Recebe o bloco de A
            req_A = comm.Irecv(A_block, source=0, tag=block_row_A)
            req_A.Wait()

            for block_col_B in range(num_blocks_B):
                col_start_B = block_col_B * block_size
                col_end_B = min(col_start_B + block_size, B_shape[1])
                
                # Buffer para receber o bloco de B
                B_block = np.empty((B_shape[0], col_end_B - col_start_B), dtype=np.float64)
                
                # Recebe o bloco de B
                req_B = comm.Irecv(B_block, source=0, tag=num_blocks_A + block_col_B)
                req_B.Wait()

                # Calcula a multiplicação do bloco
                result_block = np.dot(A_block, B_block)
                
                # Envia o resultado de volta assincronamente
                send_req = comm.Isend(result_block, dest=0, 
                                     tag=block_row_A * num_blocks_B + block_col_B)
                requests.append(send_req)

        # Aguarda a conclusão de todos os envios
        MPI.Request.Waitall(requests)

    # Processo mestre
    if rank == 0:
        final_result = np.zeros((A_shape[0], B_shape[1]), dtype=np.float64)
        send_requests = []
        result_buffers = {}

        # Envia blocos de A e B para os workers
        for block_row_A in range(num_blocks_A):
            worker = (block_row_A % (size - 1)) + 1
            
            row_start_A = block_row_A * block_size
            row_end_A = min(row_start_A + block_size, A_shape[0])
            
            # Envia bloco de A
            A_block = A_full[row_start_A:row_end_A, :].copy()
            send_requests.append(comm.Isend(A_block, dest=worker, tag=block_row_A))

        # Envia blocos de B para todos os workers
        for block_col_B in range(num_blocks_B):
            col_start_B = block_col_B * block_size
            col_end_B = min(col_start_B + block_size, B_shape[1])
            
            B_block = B_full[:, col_start_B:col_end_B].copy()
            for worker in range(1, size):
                send_requests.append(comm.Isend(B_block, dest=worker, 
                                               tag=num_blocks_A + block_col_B))

        # Prepara para receber os resultados
        for block_row_A in range(num_blocks_A):
            worker = (block_row_A % (size - 1)) + 1
            row_start_A = block_row_A * block_size
            row_end_A = min(row_start_A + block_size, A_shape[0])
            
            for block_col_B in range(num_blocks_B):
                col_start_B = block_col_B * block_size
                col_end_B = min(col_start_B + block_size, B_shape[1])
                
                result_block = np.empty((row_end_A - row_start_A, 
                                        col_end_B - col_start_B), dtype=np.float64)
                
                req = comm.Irecv(result_block, source=worker, 
                                tag=block_row_A * num_blocks_B + block_col_B)
                
                result_buffers[(block_row_A, block_col_B)] = (result_block, req, 
                                                             row_start_A, row_end_A,
                                                             col_start_B, col_end_B)

        # Aguarda a conclusão de todos os envios
        MPI.Request.Waitall(send_requests)

        # Processa os resultados à medida que chegam
        for _, (result_block, req, row_start, row_end, col_start, col_end) in result_buffers.items():
            req.Wait()
            final_result[row_start:row_end, col_start:col_end] = result_block

        print("\nResultado final da multiplicação de matrizes em blocos:")
        print(final_result)

    # Sincroniza todos os processos no final
    comm.Barrier()

if __name__ == "__main__":
    main()