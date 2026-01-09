import { Component, computed, effect, OnInit } from '@angular/core';
import { EvaluationService } from '../../services/evaluation-service';
import { ChartsModule } from '../../core/charts.module';
import { filter, map, Observable, of } from 'rxjs';
import { EChartsOption } from 'echarts';
import { AsyncPipe } from '@angular/common';

@Component({
  selector: 'app-evaluation',
  imports: [ChartsModule, AsyncPipe],
  templateUrl: './evaluation.html',
  styleUrl: './evaluation.css',
})
export class Evaluation implements OnInit {

  barChartOptions$: Observable<any>[] = [];
  heatMapOptions$: Observable<any>[] = [];

  constructor(public evaluationService: EvaluationService){
    effect(() => {
    const data = this.evaluationService.data();
    if (!data.length) return;

    this.buildChart(this.evaluationService.mrr_data(), 'MRR');
    this.buildChart(this.evaluationService.hit_rate_data(), 'Hit Rate');
    this.buildChart(this.evaluationService.accuracy_data(), 'Accuracy');
    this.buildChart(this.evaluationService.hit_rate_rank_data(), 'Hit Rate Rank');
    
    for(const key of Object.keys(this.evaluationService.heat_map_data()[0])){
      const entry = this.evaluationService.heat_map_data()[0][key];
      console.log(entry);
      for(let map = 0; map < entry.length; map++){
        let title = 'English';
        if(map % 3 === 1)
          title = 'German';
        else if(map % 3 === 2)
          title = 'Spanish';

        this.buildHeatMap(entry[map], `${key.toUpperCase()} (${title})`);
      }
    }
  });
  }

  buildChart(data: any[], title_text: string) {

    const options$ = of({
      legend: {},
      title: {
      text: title_text
      },
      tooltip: {},
      color: ['#083344', '#155E75', '#0891B2', '#C2C2C2'],
      dataset: {
        source: [
          ['Metric','English','German','Spanish','Total'],
          ...data
        ]
      },
      xAxis: {type: 'category',},
      yAxis: {},
      series: [{ type: 'bar' }, { type: 'bar' }, { type: 'bar' }, { type: 'bar' }]
    });

    this.barChartOptions$.push(options$);
  }

  buildHeatMap(inputData: any[], title_text: string) {

    const labels = [
        'CONTRADICTS',
        'NOT MENTIONED',
        'SUPPORTS',
    ];

const data = inputData
    .map(function (item) {
    return [item[1], item[0], item[2] || '-'];
});

    const options$ = of({
      legend: {},
      title: {
        text: title_text,
      },
      tooltip: {
        positin: 'top'
      },
      grid: {
        height: '50%',
        top: '10%'
      },
      color: ['#083344', '#155E75', '#0891B2'],
      xAxis: {
        type: 'category',
        name: 'Groundtruth',
        nameLocation: 'middle',
        data: labels,
        splitArea: {
          show: true
        }
      },
      yAxis: {
        type: 'category',
        name: 'Predicted',
        nameLocation: 'middle',
        data: labels,
        splitArea: {
          show: true
        }
      },
      visualMap: {
        show: false,
        min: 0,
        max: 10,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
      },
      series: 
      [
        { 
          type: 'heatmap',
          data: data,
          label: {
            show: true
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
         }
      ]
    });

    this.heatMapOptions$.push(options$);
  }

  ngOnInit(): void {
    this.evaluationService.getEvaluation();
  }

}
