"""Native PostgreSQL usage accounting, committed synchronously."""
import time
from psycopg import sql


class PostgreSQLUsageMixin:
    def update_token_counts(self,session_id,input_tokens=0,output_tokens=0,model=None,cache_read_tokens=0,
            cache_write_tokens=0,reasoning_tokens=0,estimated_cost_usd=None,actual_cost_usd=None,
            cost_status=None,cost_source=None,pricing_version=None,billing_provider=None,billing_base_url=None,
            billing_mode=None,api_call_count=0,absolute=False,source=None,task=''):
        counters=dict(input_tokens=input_tokens,output_tokens=output_tokens,cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,reasoning_tokens=reasoning_tokens,api_call_count=api_call_count)
        with self._transaction() as conn:
            self._create_on_conn(conn,session_id,source or 'unknown',model=model)
            row=self._session_on_conn(conn,session_id,lock=True)
            values={key:(value if absolute else (row.get(key) or 0)+value) for key,value in counters.items()}
            for key,value in [('estimated_cost_usd',estimated_cost_usd),('actual_cost_usd',actual_cost_usd)]:
                if value is not None: values[key]=value if absolute else (row.get(key) or 0)+value
            values.update({k:v for k,v in dict(cost_status=cost_status,cost_source=cost_source,pricing_version=pricing_version).items() if v is not None})
            if not task and not row['api_call_count'] and any(counters.values()):
                values.update({k:v for k,v in dict(model=model,billing_provider=billing_provider,billing_base_url=billing_base_url,billing_mode=billing_mode).items() if v is not None})
            self._update(conn,'sessions',values,'id',session_id)
            if not absolute and any(counters.values()):
                route=dict(session_id=session_id,model=model or (row['model'] if not task else None) or 'unknown',
                    billing_provider=billing_provider or (row['billing_provider'] if not task else None) or '',
                    billing_base_url=billing_base_url or (row['billing_base_url'] if not task else None) or '',
                    billing_mode=billing_mode or (row['billing_mode'] if not task else None) or '',task=task)
                fields={**counters,'estimated_cost_usd':estimated_cost_usd or 0,'actual_cost_usd':actual_cost_usd or 0}
                columns=[*route,*fields,'first_seen','last_seen']
                query=sql.SQL('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT(session_id,model,billing_provider,billing_base_url,billing_mode,task) DO UPDATE SET {},last_seen=excluded.last_seen').format(self.table('session_model_usage'),
                    sql.SQL(',').join(map(sql.Identifier,columns)),sql.SQL(',').join(sql.Placeholder() for _ in columns),
                    sql.SQL(',').join(sql.SQL('{}={}.{}+excluded.{}').format(sql.Identifier(key),self.table('session_model_usage'),sql.Identifier(key),sql.Identifier(key)) for key in fields))
                conn.execute(query,[*route.values(),*fields.values(),time.time(),time.time()])

    def queue_token_counts(self,session_id,**kwargs):
        self.update_token_counts(session_id,**kwargs)

    def flush_token_counts(self,timeout=5):
        return True

    def record_auxiliary_usage(self,session_id,task,**kwargs):
        self.update_token_counts(session_id,task=task,**kwargs)

    def update_session_billing_route(self,session_id,**kwargs):
        self._set_session(session_id,**kwargs)
