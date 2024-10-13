package com.khan.spark.SlidingWindows


class WindowedData(private val _input: Array[String], private val _target: String) {
  def input: Array[String] = input
  def target: String = _target
}

class EmbeddedWindowData(private val _input: Array[String], private val _target: String) {
  def input: Array[String] = input
  def target: String = _target
}