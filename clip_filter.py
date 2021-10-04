import clip
import torch
from PIL import Image
import torch.nn as nn
from multiprocessing import cpu_count

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

num_gpus = torch.cuda.device_count()
multiple_gpus = num_gpus > 1

use_jit = use_cuda and (not multiple_gpus) and '1.7.1' in torch.__version__

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.image_transform = preprocess
        self.tokenizer = clip.tokenize

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.image_transform(Image.open(row["PATH"])),
            self.tokenizer(str(row["TEXT"]), truncate=True)[0],
        )

class CLIPModel(nn.Module):
    def __init__(self, clip_model, sim_threshold):
        super().__init__()
        self.clip_model = clip_model
        self.sim_threshold = sim_threshold
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        with torch.no_grad():
            self.categories = self.clip_model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]))
            self.underaged_categories = self.clip_model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]))
            self.animal_categories = self.clip_model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]))
        self.all_categories = (self.categories, self.underaged_categories, self.animal_categories)
    
    def similarity_imgalt(self, image_tensor, text_tokens):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor).float()
            text_features = self.clip_model.encode_text(text_tokens).float()
            similarity = self.cosine_similarity(image_features, text_features).tolist()

        return image_features, similarity

    @staticmethod
    def prob(image_features, text_features):
        text_features = text_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices
    
    def probs(self, image_features, cats):
        return (CLIPModel.prob(image_features, category) for category in cats)

    def forward(self, tensors: torch.Tensor, tokens: torch.Tensor):
        dev = tensors.device
        cats = (cat.to(dev) for cat in self.all_categories)
        image_features, similarities = self.similarity_imgalt(tensors, tokens)

        probs = [self.probs(image_feature, cats) if similarity < self.sim_threshold else None \
            for image_feature, similarity in zip(image_features, similarities)]

        return image_features, similarities, probs

class CLIP:
    def __init__(self, sim_threshold):
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu", jit=use_jit)
        self.model = CLIPModel(self.clip_model, sim_threshold)

        if multiple_gpus:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)

    def preprocess_images(self, df):
        ret_image_features = []
        ret_similarity = []
        ret_probs = []
        batch_size = 256*num_gpus if use_cuda else 8
        dataset = CLIPDataset(df, self.preprocess)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        for tensors, tokens in dataloader:
            tensors, tokens = tensors.to(device), tokens.to(device)

            image_features, similarities, probs = self.model(tensors, tokens)

            ret_image_features.extend(image_features)
            ret_similarity.extend(similarities)
            ret_probs.extend(probs)
        return ret_image_features, ret_similarity, ret_probs

sim_threshold = 0.3
clip_filter = CLIP(sim_threshold)

def df_clipfilter(df):
    underaged_text = ["teen", "kid", "child", "baby"]

    img_embedding, similarities, probs = clip_filter.preprocess_images(df)
    tmp_embed = []

    df["dropped"] = False

    for i, img_embed in enumerate(img_embedding):
        if probs[i] is None: # if the similaroty didn't meet the threshold
            df.at[i, 'dropped'] = True
            continue

        nsfw_prob, underage_prob, animal_prob = probs[i]

        # get most similar categories
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            tmp_embed.append(img_embed)
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        if underage_prob[0] < 4 or underage_prob[1] < 4 or any(x in df.at[i, "TEXT"] for x in underaged_text):
            df.at[i, 'dropped'] = True
            continue

        if animal_prob[0] > 20:
            df.at[i, 'dropped'] = True
            continue
        tmp_embed.append(img_embed)
        
    df = df[df["dropped"] != True]
    df.reset_index(drop=True, inplace=True)
    return tmp_embed, df


def df_tfrecords(df, output_fname):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                int(df_image["HEIGHT"]),
                int(df_image["WIDTH"]),
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


def filter(df, out_fname, output_folder):
    # save hashes
    # df.loc[:,"hash"] = df.apply(lambda row: hashlib.md5((str(row.URL)+str(row.TEXT)).encode("utf-8")).hexdigest(), axis=1) # seems already set from gpu.py
    with open(f"{output_folder}hashes-{out_fname}.clp", "wt") as f:
        for item in df["hash"]:
            f.write(item + "\n")
    results = []
    #start0 = start = time.time()
    img_embeddings, dff = df_clipfilter(df)
    dff.to_csv(f"{output_folder}{out_fname}.csv", index=False, sep="|")

    #count results for each worker from resulting dff
    dff.loc[:,"shard"] = dff.PATH.apply(lambda x: x.split("/")[1])
    results = dff["shard"].value_counts()
    #print(f"CLIP ran in {round(time.time()-start,2)}")
    #start = time.time()
    '''
    img_embeds_sampleid = {}
    for i, img_embed_it in enumerate(img_embeddings):
        dfid_index = dff.at[i, "SAMPLE_ID"]
        img_embeds_sampleid[str(dfid_index)] = img_embed_it
    with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
        pickle.dump(img_embeds_sampleid, f)
    '''
    #print(f"Embeddings ran in {round(time.time()-start,2)}")
    #start = time.time()
    '''
    # we do not need anymore tfrecord files
    df_tfrecords(
        dff,
        f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
    )
    '''
    # save hashes
    #dff.loc[:,"hash"] = dff.apply(lambda row: hashlib.md5((str(row.URL)+str(row.TEXT)).encode("utf-8")).hexdigest(), axis=1)
    with open(f"{output_folder}hashes-{out_fname}.hsh", "wt") as f:
        for item in dff["hash"]:
            f.write(item + "\n")

    return len(dff), results
