from petastorm.unischema import Unischema, UnischemaField
from petastorm.codecs import ScalarCodec, NdarrayCodec
from pyspark.sql.types import IntegerType, StringType, FloatType

import numpy as np

BandSchema1000 = Unischema('BandSchema1000', [
   UnischemaField('index', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('timestamp', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('spatial', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('file', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('data', np.float32, (1000, 1000), NdarrayCodec(), False),
   UnischemaField('x_ul', np.float32, (), ScalarCodec(FloatType()), False),
   UnischemaField('y_ul', np.float32, (), ScalarCodec(FloatType()), False),
   UnischemaField('group_id', np.float32, (), ScalarCodec(FloatType()), False),
])

BandSchema256 = Unischema('BandSchema256', [
   UnischemaField('index', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('timestamp', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('spatial', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('file', np.string_, (), ScalarCodec(StringType()), False),
   UnischemaField('data', np.float32, (256, 256), NdarrayCodec(), False),
   UnischemaField('x_ul', np.float32, (), ScalarCodec(FloatType()), False),
   UnischemaField('y_ul', np.float32, (), ScalarCodec(FloatType()), False),
   UnischemaField('group_id', np.int32, (), ScalarCodec(IntegerType()), False),
   UnischemaField('patch_id', np.int32, (), ScalarCodec(IntegerType()), False),
])
