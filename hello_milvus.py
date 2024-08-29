# hello_milvus.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time
import os


import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
# 获取文件夹路径
directory = 'D:/pythonpro/pic02'

# 存储预处理后的图像
processed_images = []
filename_images = []
id = []
entities=[]

# 加载预训练的VGG-16模型
#model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = VGG16(weights='imagenet')
i=0
print("f-1----------------------")
print("f-2----------------------")

for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        img_path = os.path.join(directory, filename)
        # 加载图像并调整大小
        img = Image.open(img_path)
        img = img.resize((224, 224))  # 调整图像大小以适应 VGG16 模型的输入尺寸
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 添加 batch 维度
        img_array = preprocess_input(img_array)  # 预处理图像数据

        # 使用模型对图像进行编码
        vector = model.predict(img_array)
        print(i,'pic')
        # 输出向量值
        i = i+1
        entity = {
            "pk": i,
            "picfile": filename,
            #"embeddings": [1.2,1.9]
            "embeddings": vector.tolist()[0]
            # 将 NumPy 数组转换为列表
        }
        #print(vector.tolist()[0])
        entities.append(entity)
        #if i>5 :
        #    break
#print(entities)
first_embedding_of_first_entry = entities[0]['embeddings']

print("f-----------------------")
print(first_embedding_of_first_entry)
print(entities[0]['picfile'])
print("f--------------------")
print("f-----------hello milvus---------")
print("f-----------hello milvus smas---------")
print("f-----------hello milvus smas  fetch---------")
#exit(0)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 1000

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to VerctorDB"))
connections.connect("default", host="192.168.99.100", port="19530")

has = utility.has_collection("picture_search")
print(f"Does collection hello_milvus exist in VerctorDB: {has}")
if has:
    utility.drop_collection("picture_search")
    print("删除picture_search")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   INT64  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "picfile"  |    VarChar  |                  |      "a VarChar field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=1000        |  "float vector with dim 1000"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="picfile", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1000)
]

schema = CollectionSchema(fields, "this is picture search demo")

print(fmt.format("Create collection `picture_search`"))
picture_search = Collection("picture_search", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `hello_milvus`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))

insert_result = picture_search.insert(entities)
picture_search.flush()

#exit(0)
print(f"Number of entities in Milvus: {picture_search.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for hello_milvus collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024},
}

picture_search.create_index("embeddings", index)

exit(0)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `hello_milvus` into memory.
print(fmt.format("Start loading"))
picture_search.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
vectors_to_search = [first_embedding_of_first_entry]
print(vectors_to_search)
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = picture_search.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["picfile"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, picfile field: {hit.entity.get('picfile')}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `pk > 0`"))

start_time = time.time()
result = picture_search.query(expr="pk > 0", output_fields=["pk","picfile", "embeddings"])
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))



# -----------------------------------------------------------------------------
# pagination
r1 = picture_search.query(expr="pk > 0", limit=4, output_fields=["picfile"])
r2 = picture_search.query(expr="pk > 0", offset=1, limit=3, output_fields=["picfile"])
print(f"query pagination(limit=4):\n\t{r1}")
print(f"query pagination(offset=1, limit=3):\n\t{r2}")

exit(0)

# -----------------------------------------------------------------------------
# hybrid search
print(fmt.format("Start hybrid searching with `random > 0.5`"))

start_time = time.time()
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, random field: {hit.entity.get('random')}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
ids = insert_result.primary_keys

expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
print(fmt.format(f"Start deleting with expr `{expr}`"))

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

hello_milvus.delete(expr)

result = hello_milvus.query(expr=expr, output_fields=["random", "embeddings"])
print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the hello_milvus collection
print(fmt.format("Drop collection `hello_milvus`"))
#utility.drop_collection("hello_milvus")
