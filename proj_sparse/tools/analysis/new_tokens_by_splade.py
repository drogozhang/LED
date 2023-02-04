"""
Newly generated tokens.
Top-10 tokens
"""
import collections

splade_document_path = []

from peach.common import *
from peach.base import *

from peach.datasets.marco.dataset_marco_eval import DatasetMacroPassages
from transformers import AutoTokenizer
from proj_sparse.legacy_components.modeling_splade import SpladeEnocder
# from peach.enc_utils.eval_sparse import sparse_vector_to_dict

def sparse_vector_to_dict(sparse_vec, vocab_id2token, quantization_factor, dummy_token):

    idx = np.nonzero(sparse_vec)
    # then extract values:
    data = sparse_vec[idx]
    data = np.rint(data * quantization_factor).astype(int)

    dict_sparse = dict()

    for id_token, value_token in zip(idx[0], data):
        if value_token > 0:
            real_token = vocab_id2token[id_token]
            dict_sparse[real_token] = int(value_token)
    if len(dict_sparse.keys()) == 0:
        # print("empty input =>", id_)
        pass  # dict_sparse[dummy_token] = 1
        # in case of empty doc we fill with "[unused993]" token (just to fill
        # and avoid issues with anserini), in practice happens just a few times ...
    return dict_sparse

def slice_orderdict(od, top_n):
    data = [(k,v) for k, v in od.items()][:top_n]
    return data

def format_orderdict(od, top_n):
    data = [f"{k}@{v}" for k, v in od.items()][:top_n]
    s = ", ".join(data)
    return data


args = CustomArgs(
    data_dir="/relevance2-nfs/shentao/text_corpus/doc_pretrain_corpus",
    max_length=256,

    per_device_eval_batch_size=64,
    num_proc=1,

    # others
    # model_path="/home/shentao/ws/data/runtime/sparse/splade_max_distilbert",
    model_path="distilbert-base-uncased",
    sparse_emb_path="/home/shentao/ws/data/runtime/sparse_archive/debug-splade_max_distilbert-eval-notitle/sparse_retrieval/sparse_emb_dir/sparse_emb.jsonl"
)
add_title = False

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_id2token = dict((i, s) for s, i in tokenizer.get_vocab().items())

passages_dataset = DatasetMacroPassages(
    "dev", args.data_dir, None, args, tokenizer, add_title=add_title)

encoder = SpladeEnocder.from_pretrained(args.model_path).cuda()
encoder.eval()

passages_dataloader = setup_eval_dataloader(args, passages_dataset, accelerator=None)

def trans_to_ordereddict(freq_dict):
    sorted_elems = sorted(freq_dict.items(), key=lambda e:e[1], reverse=True)
    return collections.OrderedDict(sorted_elems)


curr_idx, target = 0, 100
for batch in passages_dataloader:
    batch = dict((k, v.cuda()) for k, v in batch.items())
    with torch.no_grad():
        outputs_dict = encoder(**batch, return_dict=True)

        token_embeddings = torch.log(1. + torch.relu(outputs_dict["token_embeddings"])).detach().cpu().numpy()
        sentence_embedding = outputs_dict["sentence_embedding"].detach().cpu().numpy()

        attention_mask = batch["attention_mask"].cpu().numpy()
        input_ids = batch["input_ids"].cpu().numpy()
        pids = batch["pids"].cpu().numpy()

    for idx_e in range(token_embeddings.shape[0]):
        sentence_emb = sentence_embedding[idx_e]

        attn_mask = attention_mask[idx_e]
        seq_len = int(attn_mask.sum())

        token_embs = token_embeddings[idx_e,:seq_len]
        inp_ids = input_ids[idx_e,:seq_len]

        # to show
        seq_svec = trans_to_ordereddict(sparse_vector_to_dict(sentence_emb, vocab_id2token, quantization_factor=100, dummy_token=tokenizer.unk_token))
        token_svecs = [
            trans_to_ordereddict(sparse_vector_to_dict(tk_emb, vocab_id2token, quantization_factor=100,
                                                       dummy_token=tokenizer.unk_token))
            for tk_emb in token_embs]

        token_mvecs = [collections.OrderedDict((tk, f"{val}/{seq_svec[tk]}") for tk, val in tk_svec.items())
                       for tk_svec in token_svecs]

        input_tks = tokenizer.convert_ids_to_tokens(inp_ids.tolist())

        print("=" * 30, "=" * 30)
        print("=" * 30, "=" * 30)
        print("=" * 25, int(pids[idx_e]), "=" * 25)

        for tk, tk_svec in zip(input_tks, token_mvecs):
            print(f"\"{tk}\"\t{json.dumps(format_orderdict(tk_svec, 8))}")


        curr_idx += 1

        if curr_idx > target:
            break

    if curr_idx > target:
        break





"""
sparse_vec = open(args.sparse_emb_path)
for idx_ex, example in enumerate(passages_dataset):
    sparse_example = json.loads(sparse_vec.readline())
    assert sparse_example["id"] == example["pids"]

    input_tokens = tokenizer.convert_ids_to_tokens(example["input_ids"], )

    set_input_tokens = set(input_tokens)
    set_sparse_tokens = set(sparse_example["vector"].keys())

    print("input_tokens:", input_tokens)
    print("disabled tokens:", set_input_tokens - set_sparse_tokens)
    print("new tokens:", set_sparse_tokens - set_input_tokens)

    print("="*30)
    print("="*30)
    print("="*30)

    if idx_ex >= 10:
        break
"""



































