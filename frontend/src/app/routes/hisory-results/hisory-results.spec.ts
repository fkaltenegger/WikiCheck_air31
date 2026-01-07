import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HisoryResults } from './hisory-results';

describe('HisoryResults', () => {
  let component: HisoryResults;
  let fixture: ComponentFixture<HisoryResults>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HisoryResults]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HisoryResults);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
