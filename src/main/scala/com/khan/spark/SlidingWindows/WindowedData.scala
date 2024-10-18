package com.khan.spark.SlidingWindows

import org.nd4j.linalg.api.ndarray.INDArray

class WindowedData(private val _input: Array[String], private val _target: String) {
  def input: Array[String] = input
  def target: String = _target
}

class EmbeddedWindowData(private val _input: INDArray, private val _target: INDArray) {
  def input: Array[String] = input

  def target: INDArray = _target
}