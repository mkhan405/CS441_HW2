package com.khan.spark.utils

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.SparkContext

import java.net.URI

object io {
  private val config: Config = ConfigFactory.load().resolve()
  val isProd: Boolean = sys.props.getOrElse("prod", "false").toBoolean

  /**
   * Configures the file system for reading and writing data in a Spark application based on the environment.
   *
   * @param sc The SparkContext used to access Spark's configuration and services.
   * @param baseDir The base directory URI (either local or S3) where files are stored or read from.
   *
   * @return A FileSystem instance configured according to the environment.
   *         If the application is running in a production environment, it configures
   *         the file system to use Amazon S3 with appropriate settings. Otherwise, it
   *         defaults to the local file system.
   *
   * This function first retrieves the Hadoop configuration from the provided SparkContext.
   * If the `isProd` flag is true, it sets the necessary Hadoop configuration properties
   * to interact with Amazon S3, including the implementation class and AWS credentials provider.
   * It then constructs a URI from the base directory and returns a FileSystem instance
   * configured for S3 access. If the application is not in production, it simply returns
   * the default FileSystem based on the Hadoop configuration.
   */
  def configureFileSystem(sc: SparkContext, baseDir: String): FileSystem = {
    // Get Hadoop configuration from Spark context
    val hadoopConfig = sc.hadoopConfiguration

    if (isProd) {
      // Configure S3 filesystem settings
      hadoopConfig.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      hadoopConfig.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

      val s3Uri = new URI(baseDir)
      FileSystem.get(s3Uri, hadoopConfig)
    } else {
      FileSystem.get(hadoopConfig)
    }
  }
}