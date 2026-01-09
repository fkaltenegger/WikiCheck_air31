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
  heatMapOptions$!: Observable<any>;

  constructor(public evaluationService: EvaluationService){
    effect(() => {
    const data = this.evaluationService.data();
    if (!data.length) return;

    console.log(this.evaluationService.data());

    this.buildChart(this.evaluationService.mrr_data(), 'MRR');
    this.buildChart(this.evaluationService.hit_rate_data(), 'Hit Rate');
    this.buildChart(this.evaluationService.accuracy_data(), 'Accuracy');
    this.buildChart(this.evaluationService.hit_rate_data(), 'Accurate Hit Rate');
    this.buildHeatMap(this.evaluationService.hit_rate_data(), 'Accurate Hit Rate');
  });
  }

  buildChart(data: any[], title_text: string) {

    const options$ = of({
      legend: {},
      title: {
      text: title_text
      },
      tooltip: {},
      color: ['#083344', '#155E75', '#0891B2'],
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

    const x = [
        '', 
    ];

    const y = [
        '',
    ];

const data = inputData
    .map(function (item) {
    return [item[1], item[0], item[2] || '-'];
});

    this.heatMapOptions$ = of({
      legend: {},
      title: {
      text: title_text
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
        data: x,
        splitArea: {
          show: true
        }
      },
      yAxis: {
        type: 'category',
        data: y,
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: 0,
        max: data.length,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%',
      },
      series: 
      [
        { name: 'Name',
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
  }

  ngOnInit(): void {
    this.evaluationService.getData();
  }

}
