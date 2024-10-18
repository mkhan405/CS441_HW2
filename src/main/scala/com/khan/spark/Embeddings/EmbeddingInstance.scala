package com.khan.spark.Embeddings

class EmbeddingInstance(private val _token: String, private val _vector: Array[Float])  extends Serializable {
  def token: String = _token
  def vector: Array[Float] = _vector
}
