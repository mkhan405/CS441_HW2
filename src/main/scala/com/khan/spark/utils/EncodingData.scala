package com.khan.spark.utils

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType}

object Encoding {
  // Token Encoding Register
  val registry: EncodingRegistry = Encodings.newLazyEncodingRegistry()
  // JTolkit Encoder
  val encoding: Encoding = registry.getEncoding(EncodingType.CL100K_BASE)
}