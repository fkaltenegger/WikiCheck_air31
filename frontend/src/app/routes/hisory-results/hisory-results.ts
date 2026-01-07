import { Component, OnInit, signal } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { HistoryItem } from '../../models/history.type';
import { StorageService } from '../../services/storage-service';
import { ResultItem } from '../../models/result.type';
import { Query } from '../../models/query.type';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-hisory-results',
  imports: [CommonModule],
  templateUrl: './hisory-results.html',
  styleUrl: './hisory-results.css',
})
export class HisoryResults implements OnInit{

  queryID = signal('');
  results = signal<ResultItem[]>([]);
  query = signal<any>([]);

  constructor(private route: ActivatedRoute, private store: StorageService, private router: Router) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.queryID.set(params.get('id')!);
      const results = this.store.getByQueryID(this.queryID());
      if (results)
      {
        this.results.set(results.results);
        this.query.set(results.query);
      }
    });
  }

  back(): void{
    this.router.navigate(['/history']);
  }

  getEvalBackground(evalValue: string): string {
    switch (evalValue) {
      case 'SUPPORTS':
        return 'bg-green-500';
      case 'NOT MENTIONED':
        return 'bg-gray-500';
      case 'CONTRADICTS':
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  }

}
