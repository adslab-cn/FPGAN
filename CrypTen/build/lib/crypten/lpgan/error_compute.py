import torch
import pickle
import crypten
import pickle as pkl
import crypten.communicator as comm

def run():
    crypten.init()
    rank = comm.get().get_rank()
    # 加载高精度和低精度数据
    with open("experiment/SD_BP.pkl", "rb") as f:
        SD_BP = pickle.load(f)

    with open("experiment/SG_BP.pkl", "rb") as f:
        SG_BP = pickle.load(f)

    # 提取 MPCTensor
    SD_BP_tensors = []
    SG_BP_tensors = []
    BP = []
    # a = crypten.cryptensor([5])
    # crypten.print(f"a:{a}\na_dec:{a.get_plain_text()}")
    for item in SD_BP[:1]:
        # 假设元组中的第二个元素是 MPCTensor
        SD_BP_tensor = item[1]
        crypten.print(f"Rank:{rank}\nSD_BP_tensor :{SD_BP_tensor}", in_order=True )
        SD_BP_tensors.append(SD_BP_tensor.get_plain_text())
    # print("SD_BP_tensors", SD_BP_tensors)

    for item in SG_BP:
        # 假设元组中的第二个元素是 MPCTensor
        SG_BP_tensor = item[1]
        SG_BP_tensors.append(SG_BP_tensor.get_plain_text().mean())
    # print(SG_BP_tensors)

    BP.append(SD_BP_tensors+SG_BP_tensors)
    with open('experiment/BP.pkl', 'wb') as f:
        pkl.dump(BP, f)  

    with open("experiment/BP.pkl", "rb") as f:
        BP1 = pickle.load(f)
    # print("BP1:", BP1)
    # # 计算误差
    # for high, low in zip(high_precision_data, low_precision_data):
    #     # loss_error = abs(high["true_loss"] - low["approx_loss"])
    #     # gradient_errors = [
    #     #     torch.norm(hg - lg) for hg, lg in zip(high["true_gradients"], low["approx_gradients"])
    #     # ]
    #     # print(f"FP Error: {loss_error}, BP: {gradient_errors}")


run()