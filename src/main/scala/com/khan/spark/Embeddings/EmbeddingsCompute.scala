package com.khan.spark.Embeddings

import com.khan.spark.utils.EncodingData.encoding
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.spark.broadcast.Broadcast
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.collection.Map

object EmbeddingsCompute {

  def splitAndTokenize(str: String): String = {
    str.split(" ").map(t => encoding.encode(t)).map(bpe => bpe.toArray).map(_.mkString(" ")).mkString(" ")
  }

  def generatePositionalEmbedding(tokens: Array[String], windowSize: Int, embeddingDim: Int): INDArray = {
    val result: INDArray = Nd4j.zeros(windowSize * embeddingDim)
    (0 until windowSize).foreach(pos => {
      (0 until embeddingDim by 2).foreach(i => {
        val angle: Double = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        result.putScalar(Array(pos + i), Math.sin(angle))
        if (i + 1 < embeddingDim)
          result.putScalar(Array(pos + i + 1), Math.cos(angle))
      })
    })
    result
  }

  def generateEmbedding(tokens: Array[String], tokenToEmbeddingBroadcast:Broadcast[Map[String, Array[Float]]],
                                embeddingDim: Int): INDArray = {
//    // Convert string of Integer Token IDs -> IntArrayList
//    val intTokens = new IntArrayList()
//    tokens.foreach(t => intTokens.add(t.toInt))
//
//    val decodedString = encoding.decode(intTokens)

    // For each decoded integer ID, generate embedding
    val embeddings: Array[Float] = tokens.flatMap(t => {
      val tokenList = new IntArrayList()
      tokenList.add(t.toInt)
      tokenToEmbeddingBroadcast.value.getOrElse(encoding.decode(tokenList), Array.fill(embeddingDim)(0.0f))
    })

    Nd4j.create(embeddings)
  }
}
