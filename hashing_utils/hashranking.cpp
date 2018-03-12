/*
 # Copyright 2018 Stanislav Pidhorskyi
 #
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 #  http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 # ==============================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef _WIN32
#include <intrin.h>
#define popcount32 __popcnt
#define popcount64 __popcnt64
#else
#define popcount32 __builtin_popcount
#define popcount64 __builtin_popcountll
#endif

#include <inttypes.h>

namespace py = pybind11;

inline float fdot(int n, float* __restrict sx, float* __restrict sy)
{
	float stemp = 0.0f;
	int m = n - 4;
	int i;
	for (i = 0; i < m; i += 5)
        stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                 sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

	for (; i < n; i++)
		stemp += sx[i] * sy[i];
	return stemp;
}

inline uint8_t hamming_distance32(uint32_t x, uint32_t y)
{
    uint32_t val = x ^ y;
    return (uint8_t)popcount32(val);
}

inline uint8_t hamming_distance64(uint64_t x, uint64_t y)
{
    uint64_t val = x ^ y;
    return (uint8_t)popcount64(val);
}

void to_int32_hashes(py::array_t<float, py::array::c_style> x, uint32_t* __restrict out)
{
	auto p = x.unchecked<2>();
    int w = (int)p.shape(1);
    int h = (int)p.shape(0);

	for (int i = 0; i < h; ++i)
	{
        uint32_t output = 0;
        uint32_t power = 1;
		const float* __restrict hash = p.data(i, 0);

		for (int y = 0; y < w; ++y)
		{
            output += (hash[y] > 0.0f ? power : 0);
            power *= 2;
		}
        out[i] = output;
	}
}

void to_int64_hashes(py::array_t<float, py::array::c_style> x, uint64_t* __restrict out)
{
	auto p = x.unchecked<2>();
    int w = (int)p.shape(1);
    int h = (int)p.shape(0);

	for (int i = 0; i < h; ++i)
	{
        uint64_t output = 0;
        uint64_t power = 1;
		const float* __restrict hash = p.data(i, 0);

		for (int y = 0; y < w; ++y)
		{
            output += (hash[y] > 0.0f ? power : 0);
            power *= 2;
		}
        out[i] = output;
	}
}

class HashRankingContext
{
public:
	HashRankingContext(): m_db_size(0), m_query_size(0), m_dbhashes(nullptr), m_queryhashes(nullptr), m_dist(nullptr), m_rank(nullptr), m_tmp(nullptr),m_labels_db(nullptr),
	m_labels_query(nullptr),m_labels_dbLDW(nullptr),m_labels_dbHDW(nullptr),m_labels_queryLDW(nullptr),m_labels_queryHDW(nullptr),m_relevance(nullptr),m_cumulative(nullptr),m_precision(nullptr)
	{
	}

	~HashRankingContext()
	{
		delete[] m_dbhashes;
		delete[] m_queryhashes;
		delete[] m_dist;
		delete[] m_rank;
		delete[] m_tmp;
		delete[] m_labels_db;
		delete[] m_labels_query;
		delete[] m_labels_dbLDW;
		delete[] m_labels_dbHDW;
		delete[] m_labels_queryLDW;
		delete[] m_labels_queryHDW;
		delete[] m_relevance;
		delete[] m_cumulative;
		delete[] m_precision;
	}

	enum HashStorage
	{
		HS32b,
		HS64b
	};

	enum LabelComparing
	{
		LC_equality,
		LC_and,
    LC_weighted,
	};

	void Init(int db_size, int query_size, int hs, int lc)
	{
		m_hs = (HashStorage)hs;
		m_lc = (LabelComparing)lc;
		m_db_size = db_size;
		m_query_size = query_size;
		switch(hs)
		{
		case HS32b:
			m_dbhashes = reinterpret_cast<uint8_t*>(new uint32_t[m_db_size]);
			m_queryhashes = reinterpret_cast<uint8_t*>(new uint32_t[m_query_size]);
			break;
		case HS64b:
			m_dbhashes = reinterpret_cast<uint8_t*>(new uint64_t[m_db_size]);
			m_queryhashes = reinterpret_cast<uint8_t*>(new uint64_t[m_query_size]);
			break;
		}
		m_dist = new uint8_t[m_db_size];
		m_rank = new uint32_t[m_db_size];
		m_tmp = new uint32_t[m_db_size];
		m_labels_db = new uint32_t[m_db_size];
		m_labels_query = new uint32_t[m_query_size];
		m_labels_dbLDW = new uint64_t[m_db_size];
		m_labels_dbHDW = new uint64_t[m_db_size];
		m_labels_queryLDW = new uint64_t[m_query_size];
		m_labels_queryHDW = new uint64_t[m_query_size];
		m_relevance = new float[m_db_size];
		m_cumulative = new float[m_db_size];
		m_precision = new float[m_db_size];
	}

	void LoadQueryHashes(py::array_t<float, py::array::c_style> x)
	{
		if (x.unchecked<2>().shape(0) != m_query_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		switch(m_hs)
		{
		case HS32b:
			to_int32_hashes(x, reinterpret_cast<uint32_t*>(m_queryhashes));
			break;
		case HS64b:
			to_int64_hashes(x, reinterpret_cast<uint64_t*>(m_queryhashes));
			break;
		}
	}

	void LoadDBHashes(py::array_t<float, py::array::c_style> x)
	{
		if (x.unchecked<2>().shape(0) != m_db_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		switch(m_hs)
		{
		case HS32b:
			to_int32_hashes(x, reinterpret_cast<uint32_t*>(m_dbhashes));
			break;
		case HS64b:
			to_int64_hashes(x, reinterpret_cast<uint64_t*>(m_dbhashes));
			break;
		}
	}

	void LoadQueryLabels(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> x)
	{
		if (x.unchecked<1>().shape(0) != m_query_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_query, x.unchecked<1>().data(0), 4 * m_query_size);
	}

	void LoadQueryLabelsLDW(py::array_t<uint64_t, py::array::c_style> x)
	{
		if (x.unchecked<1>().shape(0) != m_query_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_queryLDW, x.unchecked<1>().data(0), 8 * m_query_size);
	}

	void LoadQueryLabelsHDW(py::array_t<uint64_t, py::array::c_style> x)
	{
		if (x.unchecked<1>().shape(0) != m_query_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_queryHDW, x.unchecked<1>().data(0), 8 * m_query_size);
	}

	void LoadDBLabels(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> x)
	{
		if (x.unchecked<1>().shape(0) != m_db_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_db, x.unchecked<1>().data(0), 4 * m_db_size);
	}

	void LoadDBLabelsHDW(py::array_t<uint64_t, py::array::c_style> x)
	{
		if (x.unchecked<1>().shape(0) != m_db_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_dbHDW, x.unchecked<1>().data(0), 8 * m_db_size);
	}

	void LoadDBLabelsLDW(py::array_t<uint64_t, py::array::c_style> x)
	{
		if (x.unchecked<1>().shape(0) != m_db_size)
		{
			throw std::runtime_error("Size of hashes block do not match the value provided at init");
		}
		memcpy(m_labels_dbLDW, x.unchecked<1>().data(0), 8 * m_db_size);
	}

	py::array_t<uint32_t> Sort(int x)
	{
		calc_hamming_dist(x);

		int32_t count[65];
		int32_t total;
		int32_t old_count;
		int8_t key;

		for (int i = 0; i < 65; ++i)
			count[i] = 0;
		for (int y = 0; y < m_db_size; ++y)
			m_rank[y] = 0;
		for (int y = 0; y < m_db_size; ++y)
			count[m_dist[y]] += 1;
		total = 0;
		old_count = 0;
		for (int i = 0; i < 65; ++i)
		{
			old_count = count[i];
			count[i] = total;
			total += old_count;
		}
		for (int y = 0; y < m_db_size; ++y)
		{
			key = m_dist[y];
			m_tmp[y] = count[key];
			count[key] += 1;
		}
		for (int y = 0; y < m_db_size; ++y)
			m_rank[m_tmp[y]] = y;

		return py::array_t<uint32_t>(
            {m_db_size},
            {sizeof(uint32_t)},
            m_rank);
	}

	float Map()
	{
		switch(m_lc)
		{
		case LC_equality:
			return MapEqual();
			break;
		case LC_and:
			return MapAnd();
			break;
    case LC_weighted:
      return MapWeighted();
      break;
		}

		return 0.0f;
	}

	void calc_hamming_dist(int x)
	{
		switch(m_hs)
		{
		case HS32b:
		{
			uint32_t* __restrict queryhashes = reinterpret_cast<uint32_t*>(m_queryhashes);
			uint32_t* __restrict dbhashes = reinterpret_cast<uint32_t*>(m_dbhashes);

			for (int j = 0; j < m_db_size; ++j)
			{
				m_dist[j] = hamming_distance32(dbhashes[j], queryhashes[x]);
			}

			break;
		}
		case HS64b:
		{
			uint64_t* __restrict queryhashes = reinterpret_cast<uint64_t*>(m_queryhashes);
			uint64_t* __restrict dbhashes = reinterpret_cast<uint64_t*>(m_dbhashes);

			for (int j = 0; j < m_db_size; ++j)
			{
				m_dist[j] = hamming_distance64(dbhashes[j], queryhashes[x]);
			}
			break;
		}
		}
	}

private:

	float MapAnd()
	{
		float map = 0;

		float number_of_relative_docs;

		for (int q =0; q < m_query_size; ++q)
		{
			Sort(q);

			for (int i =0; i < m_db_size; ++i)
			{
				int index = m_rank[i];
				m_relevance[i] =
					((m_labels_queryLDW[q] & m_labels_dbLDW[index]) | (m_labels_queryHDW[q] & m_labels_dbHDW[index])) != 0;
			}
			m_cumulative[0] = m_relevance[0];
			for (int i = 1; i < m_db_size; ++i)
			{
				m_cumulative[i] = m_relevance[i] + m_cumulative[i-1];
			}
			number_of_relative_docs = m_cumulative[m_db_size-1];

			if (number_of_relative_docs != 0)
			{
				for (int i = 0; i < m_db_size; ++i)
				{
					m_precision[i] = m_cumulative[i] / (i+1);
				}
				float ap = fdot(m_db_size, m_precision, m_relevance);
				ap /= number_of_relative_docs;
				map += ap;
			}
		}

		map /= m_query_size;

		return map;
	}


	float MapEqual()
	{
		float map = 0.0f;

		float number_of_relative_docs;

		for (int q =0; q < m_query_size; ++q)
		{
			Sort(q);

			for (int i =0; i < m_db_size; ++i)
			{
				int index = m_rank[i];
				m_relevance[i] =
					((m_labels_query[q] == m_labels_db[index])) ? 1.0f : 0.0f;
			}
			m_cumulative[0] = m_relevance[0];
			for (int i = 1; i < m_db_size; ++i)
			{
				m_cumulative[i] = m_relevance[i] + m_cumulative[i-1];
			}
			number_of_relative_docs = m_cumulative[m_db_size-1];

			if (number_of_relative_docs != 0)
			{
				for (int i = 0; i < m_db_size; ++i)
				{
					m_precision[i] = m_cumulative[i] / (i+1);
				}

				float ap = fdot(m_db_size, m_precision, m_relevance);
				ap /= number_of_relative_docs;
				map += ap;
			}
		}

		map /= m_query_size;

		return map;
	}

  int get_mir_relavance(uint32_t query, uint32_t sample)
  {
    uint32_t same = query & sample;
    return (uint8_t) popcount32(same);
  }

  float MapWeighted()
  {
    float map = 0.0f;

    for (int q = 0; q < m_query_size; q++)
    {
      Sort(q);
      int relCount = 0;
      float ap = 0.0f;
      for (int p = 0; p < m_db_size; p ++)
      {
        int index = m_rank[p];
        uint8_t m_rel = get_mir_relavance(m_labels_query[q],m_labels_db[index]);

        if (m_rel > 0)
        {
          relCount++;
          float acg = m_rel;
          for (int n = p-1;n >= 0; n--)
          {
            index = m_rank[n];
            acg += get_mir_relavance(m_labels_query[q],m_labels_db[index]);
          }
          acg /= (p+1);

          ap += acg;
        }
      }
      if (relCount > 0)
      {
        ap /= relCount;
        map += ap;
      }
   }

    map /= m_query_size;
    return map;

  }

	HashStorage m_hs;
	LabelComparing m_lc;
	int m_db_size;
	int m_query_size;
	uint8_t* __restrict m_dbhashes;
	uint8_t* __restrict m_queryhashes;
	uint8_t* __restrict m_dist;
	uint32_t* __restrict m_rank;
	uint32_t* __restrict m_tmp;
	uint32_t* __restrict m_labels_db;
	uint32_t* __restrict m_labels_query;
	uint64_t* __restrict m_labels_dbLDW;
	uint64_t* __restrict m_labels_dbHDW;
	uint64_t* __restrict m_labels_queryLDW;
	uint64_t* __restrict m_labels_queryHDW;
	float* __restrict m_relevance;
	float* __restrict m_cumulative;
	float* __restrict m_precision;
};



PYBIND11_MODULE(_hashranking, m) {
	m.doc() = "";

	py::class_<HashRankingContext>(m, "HashRankingContext")
		.def(py::init())
		.def("Init", &HashRankingContext::Init)
		.def("LoadQueryHashes", &HashRankingContext::LoadQueryHashes)
		.def("LoadDBHashes", &HashRankingContext::LoadDBHashes)
		.def("LoadQueryLabels", &HashRankingContext::LoadQueryLabels)
		.def("LoadDBLabels", &HashRankingContext::LoadDBLabels)
		.def("LoadQueryLabelsLDW", &HashRankingContext::LoadQueryLabelsLDW)
		.def("LoadDBLabelsLDW", &HashRankingContext::LoadDBLabelsLDW)
		.def("LoadQueryLabelsHDW", &HashRankingContext::LoadQueryLabelsHDW)
		.def("LoadDBLabelsHDW", &HashRankingContext::LoadDBLabelsHDW)
		.def("calc_hamming_dist", &HashRankingContext::calc_hamming_dist)
		.def("Sort", &HashRankingContext::Sort)
		.def("Map", &HashRankingContext::Map);

	//m.def("add_circle_filled", &AddCircleFilled, py::arg("centre"), py::arg("radius"), py::arg("col"), py::arg("num_segments") = 12);
}
