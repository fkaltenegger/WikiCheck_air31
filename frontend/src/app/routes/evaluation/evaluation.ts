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

  options$: Observable<any>[] = [];

  constructor(public evaluationService: EvaluationService){
    effect(() => {
    const data = this.evaluationService.data();
    if (!data.length) return;

    console.log(this.evaluationService.data());

    this.buildChart(this.evaluationService.mrr_data(), 'MRR');
    this.buildChart(this.evaluationService.hit_rate_data(), 'Hit Rate');
    this.buildChart(this.evaluationService.accuracy_data(), 'Accuracy');
    this.buildChart(this.evaluationService.hit_rate_data(), 'Accurate Hit Rate');
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
          ['Metric','English','German','Spanish'],
          ...data
        ]
      },
      xAxis: {type: 'category',},
      yAxis: {},
      series: [{ type: 'bar' }, { type: 'bar' }, { type: 'bar' }]
    });

    this.options$.push(options$);
  }

  ngOnInit(): void {
    this.evaluationService.getData();
  }

}
