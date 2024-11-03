package com.khan.spark.Training.Listeners

import org.deeplearning4j.optimize.api.{BaseTrainingListener, TrainingListener}
import org.deeplearning4j.nn.api.Model
import org.slf4j.Logger

class ScoreLogger(iterationInterval: Int, logger: Logger) extends BaseTrainingListener with Serializable {
  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    val info = model.gradientAndScore()
    val score = info.getSecond
    val gradientNorm =  info.getFirst.gradient().norm2Number().doubleValue()
    logger.info(s"Iteration $iteration score:${score} gradient norm:${gradientNorm}")
  }
}
