ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

Compile/mainClass := Some("com.khan.spark.main")

val isProd: Boolean = sys.props.getOrElse("prod", "false").toBoolean

lazy val root = (project in file("."))
  .settings(
    name := "CS441_HW2",
    version := "1.0"
  )

ThisBuild / assemblyMergeStrategy := {
  case PathList("META-INF", "services", xs@_*) => MergeStrategy.filterDistinctLines
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case "application.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}

resolvers += "Maven Repository" at "https://mvnrepository.com/artifact"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.5.1", // % (if (isProd) "provided" else "compiled") exclude("org.apache.logging.log4j", "log4j-slf4j-impl"),
  "org.apache.spark" %% "spark-core" % "3.5.1", // % (if (isProd) "provided" else "compiled") exclude("org.apache.logging.log4j", "log4j-slf4j-impl")
  "org.apache.spark" %% "spark-mllib" % "3.5.1"
)

// Configuration dependencies
libraryDependencies += "com.typesafe" % "config" % "1.4.3"

// Logging Dependencies
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test

libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.5.9",  // Logback dependency
  "org.slf4j" % "slf4j-api" % "2.0.12"             // SLF4J dependency
  //  "org.slf4j" % "slf4j-log4j12" % "2.0.13"          // Log4j to SLF4J bridge
)

// Deep learning dependencies
libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1"
// libraryDependencies += "org.deeplearning4j" % "deeplearning4j-scaleout-api" % "1.0"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1"
libraryDependencies += "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-M2.1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nn" % "0.9.1"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
// libraryDependencies += "org.deeplearning4j" % "deeplearning4j-datasets" % "1.0.0-M2.1"