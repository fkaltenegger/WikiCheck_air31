import { Component, OnInit, signal } from '@angular/core';
import { StorageService } from '../../services/storage';
import { ResultItem } from '../../models/result.type';
import { Router } from '@angular/router';

@Component({
  selector: 'app-history',
  imports: [],
  templateUrl: './history.html',
  styleUrl: './history.css',
})
export class History implements OnInit {

  history = signal<any>([]);

  constructor(private storageService: StorageService, private router: Router){}

  ngOnInit(): void {
    this.history.set(this.storageService.getAll());
    console.log(this.history());
  }
  
  openResults(queryID: string) {
    this.router.navigate(['history-results', queryID]);
  }

}
