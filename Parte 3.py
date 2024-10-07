from mpi4py import MPI
import numpy as np

matrix_paths = r"C:/Users/carlo/OneDrive - sga.pucminas.br/Área de Trabalho/PUC/6º semestre/Sistemas Distribuídos/Trabalho open MPI/Matrizes/"
matrix_a = "Matriz5.txt"
matrix_b = "Matriz6.txt"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_matrix_from_txt(file_path):
    return np.loadtxt(file_path)

def main():
    if rank == 0:
        pathA = matrix_paths + matrix_a
        pathB = matrix_paths + matrix_b 

        A = read_matrix_from_txt(pathA)
        B = read_matrix_from_txt(pathB)

        total_rows_A = A.shape[0]
        total_cols_B = B.shape[1]
        cols_A = A.shape[1]  # Isso também é rows_B
    else:
        A = None
        B = None
        total_rows_A = None
        total_cols_B = None
        cols_A = None

    # Transmissão das dimensões para todos os processos
    total_rows_A = comm.bcast(total_rows_A, root=0)
    total_cols_B = comm.bcast(total_cols_B, root=0)
    cols_A = comm.bcast(cols_A, root=0)

    if rank != 0:
        requests = []
        results_to_send = []
        
        # Inicializa buffers para receber dados
        row_buffer = np.empty(cols_A, dtype=np.float64)
        col_buffer = np.empty(cols_A, dtype=np.float64)
        
        # Cada worker processa múltiplas linhas
        for i in range(rank - 1, total_rows_A, size - 1):
            # Recebe uma linha de A
            row_req = comm.Irecv(row_buffer, source=0, tag=i)
            row_req.Wait()
            
            result = np.zeros(total_cols_B, dtype=np.float64)
            col_requests = []
            
            # Recebe e processa as colunas de B
            for j in range(total_cols_B):
                col_req = comm.Irecv(col_buffer, source=0, tag=i*total_cols_B + j + total_rows_A)
                col_requests.append((col_req, j))
            
            # Aguarda todas as colunas e calcula os resultados
            for col_req, j in col_requests:
                col_req.Wait()
                result[j] = np.dot(row_buffer, col_buffer)
            
            # Envia o resultado de volta assincronamente
            send_req = comm.Isend(result.copy(), dest=0, tag=i)
            requests.append(send_req)
            print(f"Processo {rank} processou a linha {i}")

        # Aguarda a conclusão de todos os envios
        MPI.Request.Waitall(requests)

    # Processo mestre
    if rank == 0:
        # Inicializa a matriz de resultado final
        final_result = np.zeros((total_rows_A, total_cols_B), dtype=np.float64)
        send_requests = []
        recv_requests = []
        result_buffers = [np.empty(total_cols_B, dtype=np.float64) for _ in range(total_rows_A)]

        # Envia dados para os workers
        for i in range(total_rows_A):
            worker = (i % (size - 1)) + 1
            
            # Envia linha de A
            send_requests.append(comm.Isend(A[i].copy(), dest=worker, tag=i))
            
            # Envia colunas de B
            for j in range(total_cols_B):
                col_data = B[:, j].copy()
                send_requests.append(comm.Isend(col_data, dest=worker, tag=i*total_cols_B + j + total_rows_A))
            
            # Prepara para receber os resultados
            recv_requests.append(comm.Irecv(result_buffers[i], source=worker, tag=i))

        # Aguarda a conclusão de todos os envios
        MPI.Request.Waitall(send_requests)

        # Aguarda todos os recebimentos e constrói o resultado final
        for i, req in enumerate(recv_requests):
            req.Wait()
            final_result[i] = result_buffers[i]

        print("\nResultado final da multiplicação paralela:")
        print(final_result)

if __name__ == "__main__":
    main()
