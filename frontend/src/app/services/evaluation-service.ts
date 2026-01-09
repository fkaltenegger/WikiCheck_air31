import { HttpClient } from '@angular/common/http';
import { effect, Injectable, signal } from '@angular/core';
import { map, Observable, of } from 'rxjs';
import { Metrics } from '../models/metrics.type';

@Injectable({
  providedIn: 'root',
})
export class EvaluationService {

  data = signal<Metrics[]>([]);
  loading = signal(false);
  error = signal<string | null>(null);

  mrr_data = signal<any>([]);
  hit_rate_data = signal<any>([]);
  accuracy_data = signal<any>([]);
  hit_rate_rank_data = signal<any>([]);
  heat_map_data = signal<any>([]);

  constructor(private http: HttpClient) {
    effect(() => {
      if (this.data().length) 
      {
        this.buildDataMap('mrr');
        this.buildDataMap('hit_rate');
        this.buildDataMap('accuracy');
        this.buildDataMap('hit_rate_rank');
        this.buildHeatMapData();
      }
    })
  }

  buildHeatMapData(){
    const dataMap: Record<string, any[]> = {
      'tf-idf_ce_on' : [],
      'tf-idf_ce_off' : [],
      'mbert_ce_on' : [],
      'mbert_ce_off' : [],
      'sbert_ce_on' : [],
      'sbert_ce_off' : [],
    };

    for(const entry of this.data()){
      let numbers = [];
      for(let x = 0; x < 3; x++){
        for(let y = 0; y < 3; y++ )
        {
          numbers.push([x, y, entry.heat_map[x][y]]);
        }
      }
      const key = `${entry.metric}_${entry.ce ? 'ce_on' : 'ce_off'}`;
      dataMap[key].push(numbers);
    }

    this.heat_map_data.update(values => [...values, dataMap]);
  }

  buildDataMap(criteria: keyof Metrics){
    const dataMap: Record<string, any[]> = {
      'tf-idf_ce_on' : ['TF-IDF_CE_on'],
      'tf-idf_ce_off' : ['TF-IDF_CE_off'],
      'mbert_ce_on' : ['MBERT_CE_on'],
      'mbert_ce_off' : ['MBERT_CE_off'],
      'sbert_ce_on' : ['SBERT_CE_on'],
      'sbert_ce_off' : ['SBERT_CE_off'],
    };

    for(const entry of this.data()) {
      const key = `${entry.metric}_${entry.ce ? 'ce_on' : 'ce_off'}`;
      dataMap[key].push(entry[criteria]);
    }

    for (const key in dataMap) {
      const sum = dataMap[key]
        .slice(1)
        .reduce((acc, val) => acc + val, 0);

      const avg = sum / (dataMap[key].length - 1);

      dataMap[key].push(avg);
    }

    if(criteria === 'mrr') 
      this.mrr_data.update(values => [...values, ...Object.values(dataMap)]);
    else if(criteria === 'hit_rate') 
      this.hit_rate_data.update(values => [...values, ...Object.values(dataMap)]);
    else if(criteria === 'accuracy') 
      this.accuracy_data.update(values => [...values, ...Object.values(dataMap)]);
    else if(criteria === 'hit_rate_rank') 
      this.hit_rate_rank_data.update(values => [...values, ...Object.values(dataMap)]);
  }

  getEvaluation() {
    this.loading.set(true);
    this.http.get<any>('http://localhost:8000/evaluation').subscribe({
      next: res => {
      const metricsList: Metrics[] = [];

      for (const metricKey in res) {
        const metric = res[metricKey];

        for (const ceKey in metric) {
          const ce = true ? ceKey === 'ce_True' : false;
          const ceData = metric[ceKey];

          for (const langKey in ceData) {
            const values = ceData[langKey];

            metricsList.push({
              metric: metricKey,
              ce: ce,
              language: langKey,
              mrr: values.mrr,
              hit_rate: values.hit_rate,
              accuracy: values.accuracy,
              hit_rate_rank: values.hit_rate_rank,
              heat_map: values.heat_map
            });
          }
        }
      }
      this.data.set(metricsList);
      this.loading.set(false);
    },
    error: () => {
      this.error.set('Failed to load metrics');
      this.loading.set(false);
    }
    })
  }  
}
