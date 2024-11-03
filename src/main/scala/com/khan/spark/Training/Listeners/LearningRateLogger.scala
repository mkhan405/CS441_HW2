package com.khan.spark.Training.Listeners

import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.Logger

class LearningRateLogger extends BaseTrainingListener {
  override def onEpochEnd(model: Model): Unit = {

  }
}
