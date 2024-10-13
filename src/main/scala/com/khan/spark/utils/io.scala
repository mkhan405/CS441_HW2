package com.khan.spark.utils

import com.typesafe.config.{Config, ConfigFactory}

object io {
  private val config: Config = ConfigFactory.load().resolve()

  def getResourcePath(path: String): String = {
    val isProd: Boolean = sys.props.getOrElse("prod", "false").toBoolean

    if (!isProd) {
      getClass.getResource(path).toString
    } else {
      val s3URI = config.getString("resources.s3URI")
      s3URI + path
    }

  }
}