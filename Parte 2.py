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

        # Calcula como dividir o trabalho
        total_rows = A.shape[0]
        rows_per_process = total_rows // (size - 1) if size > 1 else total_rows
        remainder = total_rows % (size - 1) if size > 1 else 0

        # Distribui porções da matriz A para os processos trabalhadores
        for i in range(1, size):
            start_row = (i - 1) * rows_per_process + min(i - 1, remainder)
            end_row = start_row + rows_per_process + (1 if i - 1 < remainder else 0)
            Atemp = A[start_row:end_row]
            comm.Send(Atemp, dest=i, tag=i)
    else:
        A = None
        B = None

    # Transmite as dimensões das matrizes para todos os processos
    A_shape = comm.bcast(A.shape if rank == 0 else None, root=0)
    B_shape = comm.bcast(B.shape if rank == 0 else None, root=0)

    if rank != 0:
        # Processos trabalhadores recebem sua porção de A
        rows_per_process = A_shape[0] // (size - 1)
        remainder = A_shape[0] % (size - 1)
        local_rows = rows_per_process + (1 if rank - 1 < remainder else 0)
        A = np.empty((local_rows, A_shape[1]))
        comm.Recv(A, source=0, tag=rank)

    # Transmite a matriz B para todos os processos
    B = comm.bcast(B if rank == 0 else None, root=0)

    if rank != 0:
        # Processos trabalhadores realizam multiplicação local
        local_result = np.dot(A, B)
        # Envia resultados de volta para o processo mestre
        comm.Send(local_result, dest=0, tag=rank)

    if rank == 0:
        # Processo mestre coleta e monta o resultado final
        final_result = np.zeros((A.shape[0], B.shape[1]))
        
        for i in range(1, size):
            start_row = (i - 1) * rows_per_process + min(i - 1, remainder)
            end_row = start_row + rows_per_process + (1 if i - 1 < remainder else 0)
            rows = end_row - start_row
            
            local_result = np.empty((rows, B.shape[1]))
            comm.Recv(local_result, source=i, tag=i)
            final_result[start_row:end_row] = local_result

        print("\nResultado final da multiplicação paralela:")
        print(final_result)

if __name__ == "__main__":
    main()
