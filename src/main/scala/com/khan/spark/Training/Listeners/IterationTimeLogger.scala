package com.khan.spark.Training.Listeners

import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.Logger

class IterationTimeLogger(logger: Logger) extends BaseTrainingListener with Serializable {
  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
  }
}
