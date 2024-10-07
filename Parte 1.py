from mpi4py import MPI
import numpy as np

matrix_paths =  r"Matrizes/"
matrix_a = "Matriz5.txt"
matrix_b = "Matriz6.txt"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Função para ler uma matriz de um arquivo txt
def read_matrix_from_txt(file_path):
    return np.loadtxt(file_path)

def main():
    if rank == 0:
        pathA = matrix_paths + matrix_a
        pathB = matrix_paths + matrix_b

        A = read_matrix_from_txt(pathA)
        B = read_matrix_from_txt(pathB)

    else:
        A = None
        B = None

    # Broadcast das matrizes A e B para todos os processos
    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)

    total_rows = A.shape[0]
    rows_per_process = total_rows // (size - 1)
    remainder = total_rows % (size - 1)

    # Processo diferente de rank 0 realiza multiplicação de sua parte
    if rank != 0:
        start_row = (rank - 1) * rows_per_process + min(rank - 1, remainder)
        end_row = start_row + rows_per_process + (1 if rank - 1 < remainder else 0)
        
        # Cálculo local da multiplicação
        local_result = np.dot(A[start_row:end_row], B)
        
        # Envio assíncrono dos resultados parciais
        req_send = comm.Isend(local_result, dest=0, tag=rank)
        req_send.Wait()  # Aguarda conclusão do envio

        print(f"Processo {rank} processou linhas de {start_row} a {end_row-1}")

    # Processo 0 recebe os resultados de forma assíncrona
    if rank == 0:
        final_result = np.zeros((A.shape[0], B.shape[1]))
        requests = []

        # Receber os resultados parciais de todos os outros processos
        for i in range(1, size):
            start_row = (i - 1) * rows_per_process + min(i - 1, remainder)
            end_row = start_row + rows_per_process + (1 if i - 1 < remainder else 0)
            rows = end_row - start_row
            
            local_result = np.empty((rows, B.shape[1]))

            # Recebimento assíncrono
            req_recv = comm.Irecv(local_result, source=i, tag=i)
            requests.append((req_recv, start_row, end_row, local_result))

        # Aguardando as requisições de recebimento serem finalizadas
        for req, start_row, end_row, local_result in requests:
            req.Wait()  # Espera a conclusão da recepção
            final_result[start_row:end_row] = local_result

        print("\nResultado final da multiplicação paralela:")
        print(final_result)

if __name__ == "__main__":
    main()
